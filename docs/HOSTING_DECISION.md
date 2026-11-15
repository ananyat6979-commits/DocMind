# Release hosting decision

Publish source, a reproducible container and a static recorded-evidence explorer on GitHub Pages. Do not expose the uploadable UI/API service publicly for this release. A private screen-shared deployment remains an interview option, not an availability promise.

The static site loads already-public synthetic fixtures, measured outputs, source quotes and recorded hold states. It contains no private PDFs, human annotation exports, provider credential, upload route or live inference. Failed and unserved cases remain inspectable. It is a recorded replay, not a live document-answer service. Static delivery avoids model cold starts but is not a zero-attack-surface or uptime guarantee.

GitHub documents Pages as available for public repositories on GitHub Free. Hugging Face's current documentation says creating Gradio or Docker compute Spaces requires a paid plan even though CPU Basic itself has no hourly charge. Therefore that route is incompatible with the zero-spend requirement. The full deployment's confined parser and PostgreSQL option also require their documented architecture; do not replace them with trusted-input native parsing merely to fit a hosting service.

The repository's immutable image tags, exact digests, build inventory and provenance support technical inspection. Signing identifies a build origin; it does not establish model safety, dependency freedom from defects or commercial rights in training data. Personal credentials and parser isolation remain necessary for actual deployments. No new hosted evaluation calls or paid hosting are authorized.

The three-sentence README, actual 75-second UI capture, frozen results, methods and failure analysis are the primary presentation. There is no defensible universal statistic for how long recruiters inspect a repository or which artifact guarantees an interview. The aim is to make the evidence easy to inspect, not to assert hiring outcomes.

The exact-match confound is already reported separately. The original calibrated-policy result remains a result under its EM-defined error metric; it does not establish factual risk. Replacing that metric with human correctness requires suitable new labels and a new calibration study. Verbose answers or disagreeing human subsets do not justify silently rewriting historical outcomes.

The own-document source-confirmation export now supports a frozen 17-question PDF cohort and two paired DOCX checks. The actual generated-answer and claim-support review remains separate. This small, source-confirmed task subset is not representative multilingual validation, and synthetic proposals are not human-authored questions.

Sources checked for this decision:

- [GitHub Pages publishing sources and plan availability](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site)
- [Hugging Face Spaces overview](https://huggingface.co/docs/hub/spaces-overview)
- [GitHub public-repository artifact attestations](https://docs.github.com/en/actions/how-tos/secure-your-work/use-artifact-attestations/use-artifact-attestations)
