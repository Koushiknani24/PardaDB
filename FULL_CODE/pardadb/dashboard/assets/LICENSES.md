# Third-party assets used by the landing page

All files below are stored locally so PardaDB runs fully offline. Nothing is loaded from the internet at runtime.

| File | What | Source | Author | Licence |
|---|---|---|---|---|
| `fonts/pixelify-sans-400.woff2`, `fonts/pixelify-sans-700.woff2` | Pixelify Sans, Latin subset (headings) | npm `@fontsource/pixelify-sans@5.0.4` (https://cdn.jsdelivr.net/npm/@fontsource/pixelify-sans@5.0.4/files/), upstream https://github.com/eifetx/Pixelify-Sans | The Pixelify Sans Project Authors (Stefie Justprince) | SIL Open Font License 1.1, see `licenses/OFL-PixelifySans.txt` |
| `fonts/silkscreen-400.woff2`, `fonts/silkscreen-700.woff2` | Silkscreen, Latin subset (logo, buttons, numbers) | npm `@fontsource/silkscreen@5.0.8` (https://cdn.jsdelivr.net/npm/@fontsource/silkscreen@5.0.8/files/), upstream https://github.com/googlefonts/silkscreen | The Silkscreen Project Authors (Jason Kottke) | SIL Open Font License 1.1, see `licenses/OFL-Silkscreen.txt` |
| inline SVG icons in `landing.html` (arrow-right, play, copy, search, users, shield, lock, git-commit, check, book-open, server, zap) | Pixelarticons, path data inlined | npm `pixelarticons@1.8.1` (https://github.com/halfmage/pixelarticons) | Gerrit Halfmann | MIT, see `licenses/MIT-pixelarticons.txt` |
| `favicon.svg` and the other pixel SVGs inlined in `landing.html` (logo, step icons, role avatars) | Hand-made pixel art | Team AVNI | Team AVNI | Same licence as PardaDB |

The fonts are unmodified files as published by Fontsource (Latin subset). The OFL allows bundling and redistribution with software; the font names are not used for any modified version.
