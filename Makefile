# Build, preview, check, and publish the book.
#   make build     build _site/
#   make dev       rebuild on change and serve at http://localhost:4400
#   make check     build, check every Chinese page against its English one,
#                  then load every page and report figure, script, and layout errors
#   make deploy    build, check, and publish to https://changkun.de/bobook/
#   make preview   publish the current state even if some chapters have
#                  errors (they show inline), for looking at work in progress

HOST ?= changkunde
DEST ?= /www/changkun.de/bobook/

# Builds every edition: en/ and zh/. A strict build fails if a Chinese page
# is missing (it would fall back to English).
build:
	npm run build

dev:
	npm run dev

check: build
	@node tools/zh-check.ts > /dev/null || { node tools/zh-check.ts | grep -v '^✓'; exit 1; }
	node tools/shots.ts --all

deploy: check
	rsync -az --delete _site/ $(HOST):$(DEST)
	@echo "published to https://changkun.de/bobook/"

preview:
	node src/build.ts --lenient
	rsync -az --delete _site/ $(HOST):$(DEST)
	@echo "preview published to https://changkun.de/bobook/"

.PHONY: build dev check deploy preview
