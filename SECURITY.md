# Security

## Supported deployment model

DocMind v2 is a single-user or shared pilot application. It is not a
multi-tenant service and must not process confidential documents because
retrieved passages are sent to Groq.

## Secrets

Store `GROQ_API_KEY`, `DOCMIND_API_KEY`, `DOCMIND_APP_PASSWORD`, and
`DOCMIND_REVIEWER_PIN` only in `.env` or host secret settings. Never commit,
paste, or include them in archives. Revoke any exposed key immediately.

## Reporting a vulnerability

Open a private security advisory or contact the repository owner with the
