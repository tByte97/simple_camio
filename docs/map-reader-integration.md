# Map reader integration: preparation status

The reader resolves the existing maker's loose-file export relative to its model
JSON, passes resolved paths to CamIO in memory, and integrates those paths with
the existing Piper narration generator. The maker is unchanged.

This branch starts at upstream `auto-narration` commit
`d69e05e89bb0802dfbf82771614d7be1107acc4e`. On 2026-10-19, GitHub's branches API
confirmed that this was still the latest TTS branch and that upstream `master`
was `d5abc575af608ae3f66cafe95c5b06b8919dd152`. There were no newer upstream
commits to download. The local Git fetch itself was unavailable in the sandbox.

## Suggested PR

Title: Resolve existing maker exports in the reader and TTS generator

Base: `samaranin/simple_camio:auto-narration`. Target `master` after the TTS
branch is merged there; otherwise the PR also includes the upstream TTS work.

Description:

> Maker exports contain `models/<name>/...` paths that stop working when users
> move the map directory. Resolve those paths against the exported JSON without
> changing its bytes, infer zone identity from existing filenames/navigation,
> and share resolution with offline narration generation and runtime loading.
> Add reader preflight checks and integration tests against the unchanged maker.

## Validation

The current local working tree passed 178 tests with one skipped test using
`venv-tts/bin/python -m pytest -q`. This run includes pre-existing local macOS
camera changes and a local BlackMap, which are outside the reader commit.
Pyglet printed an interpreter-exit traceback because the sandbox has no display;
pytest returned exit code 0. Hardware playback was not tested.

A clean archive of the reader commit, without local camera changes, maps or a
sibling maker checkout, passed 165 tests with nine maker integration tests
skipped. Exit code was 0, with the same Pyglet display-related exit traceback.

A local 42-zone BlackMap export passed strict preflight, and repeating offline
generation reported `nothing to generate`. Local examples, generated narration,
voice models, and test-copy folders are not part of this source change.

## Review findings deferred by the user

This is a prepared implementation, with compatibility work remaining before it
should be called ready to merge:

- **High:** Runtime image-size validation rejects the bundled legacy CnapMap
  (720x960 mask, 720x1280 template), although the old loader accepted it.
- **High:** For missing narration outside `tts/`, runtime synthesis writes a
  generated path but playback still references the original missing path.
- **Medium:** Offline generation can synthesize a new `tts/` recording and
  rewrite paths even when a usable recording already exists outside `tts/`.
- **Medium:** New maker integration tests require a sibling checkout named
  `simple-camio-map-editor` and Node.js; they skip in a standalone checkout.
- **Compatibility:** An explicitly blank `mapDescriptionText` disables an
  existing legacy description recording as well as a maker placeholder.

These findings came from an independent read-only review and local checks.
The user requested preparing existing changes only, so no fixes were applied.
Physical camera tracking and Raspberry Pi playback still require device tests.
