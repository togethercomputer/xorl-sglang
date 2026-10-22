FROM lmsysorg/sglang:latest

# CVE-2026-111943: refresh the OS trust store (removes the distrusted emSign roots).
RUN apt-get update && apt-get install -y --only-upgrade ca-certificates && rm -rf /var/lib/apt/lists/* \
    && dpkg --compare-versions "$(dpkg-query -W -f='${Version}' ca-certificates)" ge 20260816 \
    || { echo "ca-certificates is older than 20260816" >&2; exit 1; }

COPY serve /usr/bin/serve
RUN chmod 777 /usr/bin/serve

ENTRYPOINT [ "/usr/bin/serve" ]
