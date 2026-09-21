# Face analyzer UI redesign

## Intent

Make the application feel like a polished analysis workspace for serious computer-vision work while preserving all existing inference, gallery, webcam, and theme behavior.

## Audience and job

The primary audience is a technical operator who uploads an image, selects analysis features, reviews detected faces, and exports results. The first viewport must explain the task, expose the input path, and keep advanced model controls available without overwhelming the operator.

## Design direction

- Preserve the existing theme chooser, but make every theme share the same information hierarchy and accessible interaction rules.
- Treat “Optical Bench” as the default product direction: dark optical-lab surfaces, teal focus accents, quiet metadata, and restrained borders.
- Use sentence case for user-facing labels. Use Material icons for actions, never icon-only emoji buttons without accessible labels.
- Keep results visual-first: source or annotated image, summary, face details, then exports and advanced actions.
- Make face information available without hover. Hover remains an enhancement, not the only access path.
- Put sensitive-model limitations near the results, not only in advanced settings.

## Invariants

- Exactly one face detector runs per frame.
- Active model sets continue to control inference cost and output.
- Gallery persistence and identity matching behavior remain unchanged.
- Live callbacks keep using cached, thread-safe tracker and fusion objects.
- Existing theme names and their visual differences remain available.

## Acceptance

- Desktop first viewport has clear title, input tabs, upload/camera control, and readable contrast.
- Mobile viewport does not let the sidebar obscure or clip the primary workflow.
- Results have a stable hierarchy and visible face details independent of hover.
- Browser console stays clean; source tests and the full test suite pass.
