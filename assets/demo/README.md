# Real workflow capture

These are browser screenshots of the actual Streamlit UI and authenticated API, captured on free CPU CI at the source revision in `manifest.json`. They use original synthetic English/Hindi documents, pinned E5 retrieval and an offline extractive reader. They do not measure generative answer quality or production availability. No human review decision was submitted, and the temporary credentials are masked in the UI and discarded with the isolated service.

[Watch the 75-second workflow](docmind-workflow.gif). The numbered headings and footer in the presentation frames were added outside the UI pixels. They are explanatory captions, not app labels. `captions.json` stores them separately. Real in-app labels, including draft, pending, Human review and Choose a decision, remain unchanged.

Plain, unframed browser screenshots:

1. [Authenticated library](ui-1.png)
2. [Question entry](ui-2.png)
3. [Unverified draft and quoted evidence](ui-3.png)
4. [Expanded retrieved passage](ui-4.png)
5. [Unvalidated Hindi response held](ui-5.png)
6. [Authorized reviewer with no decision selected](ui-6.png)

`frame-*.png` are the separately captioned presentation versions. The manifest records hashes for both forms and the GIF. No screenshot was synthesized or redrawn. Current UI behavior is covered by the capture workflow and regression tests; reader measurements identify their own source revisions separately.
