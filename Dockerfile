# The hosted SozoGraph passport: MCP at /mcp, the visa office, the REST mirror.
#
#   docker build -t sozograph .
#   docker run -p 8080:8080 -e SOZOGRAPH_ISSUER=https://passport.example.com \
#     -e SOZOGRAPH_HOLDER_SECRET=... -e SOZOGRAPH_SERVER_KEY=... \
#     -e SOZOGRAPH_PASSPORT=firestore://PROJECT/passports/HOLDER sozograph
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=8080 \
    SOZOGRAPH_PASSPORT=/data/passport.json
WORKDIR /app
COPY pyproject.toml README.md LICENSE CHANGELOG.md MIGRATING.md ./
COPY src ./src
COPY schemas ./schemas
RUN pip install --no-cache-dir ".[server]" && useradd --create-home sozo && mkdir -p /data && chown sozo /data
USER sozo
EXPOSE 8080
CMD ["sh", "-c", "sozograph serve --http --host 0.0.0.0 --port ${PORT} --passport ${SOZOGRAPH_PASSPORT} --issuer ${SOZOGRAPH_ISSUER}"]
