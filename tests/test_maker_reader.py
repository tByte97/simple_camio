"""Compatibility with the unchanged maker's exporter and exact RGB rasterizer."""
import base64
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import wave

import cv2 as cv
import numpy as np
import pytest

from src.map_reader import read_map, resolve_model, validate_images
from src.tts import generate_audio

MAKER = Path(__file__).resolve().parents[2] / 'simple-camio-map-editor'


@pytest.fixture
def maker_export(tmp_path):
    if not (MAKER / 'src/export-model.js').is_file() or shutil.which('node') is None:
        pytest.skip('Maker compatibility tests require the sibling maker checkout and Node.js')
    script = '''
      const p = require('./test/fixtures/project-v1.json');
      p.name = 'OriginalName';
      p.audio = {...require('./src/project.js').DEFAULT_AUDIO};
      p.zones[0].label = 'Сходи';
      p.zones.push({id:'zone-2',label:'Сходи',color:[76,217,76],
                    polygon:[[45,0],[75,0],[75,30],[45,30]]});
      const document = require('./src/export-model.js').toModel(p, {
        rasterPath:'models/OriginalName/OriginalName.png',
        templatePath:'models/OriginalName/template.jpg',
        audioDir:'models/OriginalName/Audio'});
      const r = require('./src/raster.js').rasterize(p);
      console.log(JSON.stringify({document,width:r.width,height:r.height,
                                 rgba:Buffer.from(r.data).toString('base64')}));
    '''
    data = json.loads(subprocess.check_output(['node', '-e', script], cwd=MAKER))
    folder = tmp_path / 'DifferentFolderName'
    folder.mkdir()
    path = folder / 'OriginalName.json'
    path.write_text(json.dumps(data['document'], ensure_ascii=False, indent=2), encoding='utf-8')
    rgba = np.frombuffer(base64.b64decode(data['rgba']), np.uint8).reshape(data['height'], data['width'], 4)
    cv.imwrite(str(folder / 'OriginalName.png'), rgba[:, :, :3][:, :, ::-1])
    cv.imwrite(str(folder / 'template.jpg'), np.full(rgba.shape[:2], 128, np.uint8))
    return path, data['document']


def wav(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), 'wb') as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(22050)
        stream.writeframes(b'\0\0' * 100)


def test_original_export_rebases_without_new_schema(maker_export, tmp_path, monkeypatch):
    path, original = maker_export
    monkeypatch.chdir(tmp_path)
    before = path.read_bytes()
    model = read_map(path)
    validate_images(model, strict=True)
    assert 'mapPackageVersion' not in original
    assert 'id' not in original['model']['hotspots'][0]
    assert model['filename'] == str(path.parent / 'OriginalName.png')
    assert model['template_image'] == str(path.parent / 'template.jpg')
    assert [h['id'] for h in model['hotspots']] == ['zone-1', 'zone-2']
    assert model['hotspots'][0]['audioDescription'] == str(path.parent / 'Audio/tts/zone-1.wav')
    assert 'map_description' not in model
    assert path.read_bytes() == before


def test_blank_maker_description_does_not_play_a_stale_recording(maker_export):
    path, document = maker_export
    wav(path.parent / 'Audio/tts/map_description.wav')
    assert 'map_description' not in resolve_model(document, path)


def test_generation_and_move_leave_export_json_byte_identical(maker_export, tmp_path, monkeypatch, stub_piper, stub_voice):
    path, _ = maker_export
    monkeypatch.chdir(tmp_path)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    run = lambda p: generate_audio.generate_for_model(p, piper_bin=str(stub_piper), voice='test_voice', voices_dir=stub_voice)
    assert run(path) == 0
    outputs = sorted((path.parent / 'Audio/tts').glob('*.wav'))
    assert [p.name for p in outputs] == ['zone-1.wav', 'zone-2.wav']
    stamps = [p.stat().st_mtime_ns for p in outputs]
    target = tmp_path / 'MovedToPi'
    shutil.move(str(path.parent), target)
    moved = target / path.name
    assert run(moved) == 0
    assert [p.stat().st_mtime_ns for p in sorted((target / 'Audio/tts').glob('*.wav'))] == stamps
    assert hashlib.sha256(moved.read_bytes()).hexdigest() == before
    assert all(Path(h['audioDescription']).is_file() for h in read_map(moved)['hotspots'])


def test_optional_description_generates_at_the_existing_maker_path(maker_export, stub_piper, stub_voice):
    path, document = maker_export
    document['model']['mapDescriptionText'] = 'Опис карти'
    path.write_text(json.dumps(document, ensure_ascii=False), encoding='utf-8')
    before = path.read_bytes()
    assert generate_audio.generate_for_model(path, piper_bin=str(stub_piper), voice='test_voice', voices_dir=stub_voice) == 0
    assert Path(read_map(path)['map_description']).is_file()
    assert path.read_bytes() == before


def test_empty_audio_path_uses_navigation_id_and_unique_label(maker_export):
    path, document = maker_export
    h = document['model']['hotspots'][0]
    h['audioDescription'] = ''
    h['textDescription'] = 'Перша зона'
    target = path.parent / 'Recordings/Перша_зона.wav'
    wav(target)
    assert resolve_model(document, path)['hotspots'][0]['audioDescription'] == str(target)
    target.unlink()
    assert resolve_model(document, path)['hotspots'][0]['audioDescription'] == str(path.parent / 'Audio/tts/zone-1.wav')


def test_repeated_labels_and_duplicate_stems_are_rejected(maker_export):
    path, document = maker_export
    label = path.parent / 'Clips/Сходи.wav'
    wav(label)
    with pytest.raises(ValueError, match='Repeated zone label'):
        resolve_model(document, path)
    label.unlink()
    wav(path.parent / 'A/zone-1.wav')
    wav(path.parent / 'B/zone-1.wav')
    with pytest.raises(ValueError, match='Ambiguous audio'):
        resolve_model(document, path)


def test_no_fallback_to_another_installed_map(maker_export, tmp_path):
    path, document = maker_export
    app = tmp_path / 'application'
    wav(app / 'models/OriginalName/Audio/tts/zone-1.wav')
    model = resolve_model(document, path, app_root=app)
    assert model['hotspots'][0]['audioDescription'] == str(path.parent / 'Audio/tts/zone-1.wav')
    assert not Path(model['hotspots'][0]['audioDescription']).exists()


def test_sizes_and_unknown_mask_colors_fail_preflight(maker_export):
    path, document = maker_export
    model = resolve_model(document, path)
    cv.imwrite(str(path.parent / 'template.jpg'), np.zeros((4, 4), np.uint8))
    with pytest.raises(ValueError, match='differs from template'):
        validate_images(model, strict=True)
    cv.imwrite(str(path.parent / 'template.jpg'), np.zeros((80, 100), np.uint8))
    mask = cv.imread(model['filename'])
    mask[0, 0] = (1, 2, 3)
    cv.imwrite(model['filename'], mask)
    with pytest.raises(ValueError, match='undeclared colors'):
        validate_images(model, strict=True)


def test_identity_conflicts_and_escaping_symlinks_are_rejected(maker_export, tmp_path):
    path, document = maker_export
    h = document['model']['hotspots'][0]
    original = h['audioDescription']
    h['audioDescription'] = 'models/OriginalName/Audio/tts/zone-99.wav'
    with pytest.raises(ValueError, match='Conflicting zone identity'):
        resolve_model(document, path)
    h['audioDescription'] = original
    outside = tmp_path / 'outside'
    outside.mkdir()
    (path.parent / 'Audio').symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match='escapes the map directory'):
        resolve_model(document, path)
