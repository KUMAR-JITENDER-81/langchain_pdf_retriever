# Security policy

## Reporting a vulnerability

Please do not publish credentials, private PDFs, or exploit details in a public issue.
Contact the repository owner privately and include the affected endpoint, reproduction
steps, impact, and a suggested mitigation when available.

## Deployment checklist

- Set a strong `API_AUTH_TOKEN`; never commit `backend/.env`.
- Put the API behind HTTPS and an authenticated reverse proxy.
- Use per-user authorization and storage isolation before serving multiple users.
- Keep the Python, container, OCR, and JavaScript dependencies patched.
- Restrict upload, page, OCR, storage, and request-rate limits for the available hardware.
- Treat uploaded PDFs and extracted text as untrusted content.
- Review the privacy policy of any external OCR, embedding, or generation provider.
- Scan uploads with an antivirus service when accepting files from untrusted users.
- Back up `uploads`, `data`, and `chroma_db` together so metadata and vectors remain consistent.

The included bearer token and in-memory rate limiter are suitable for a private,
single-instance deployment. Public or multi-tenant deployments should use an identity
provider, tenant-aware database records, a durable job queue, and centralized rate limits.
