#!/usr/bin/env bash
#
# ship.sh -- unzip a delivery over this repo, test it, commit it, push it.
#
#   ./ship.sh ~/Downloads/chrischart-volume-pace.zip "what changed"
#   ./ship.sh ~/Downloads/chrischart-volume-pace.zip          # asks for a message
#   ./ship.sh --no-push ~/Downloads/thing.zip "message"       # stop before pushing
#   ./ship.sh --dry-run ~/Downloads/thing.zip                 # unpack and test only
#
# Railway deploys from GitHub, so unzipping without pushing changes nothing
# that is actually running. That is the whole reason this exists.
#
# WHAT IT REFUSES TO DO
#
#   - commit anything outside liqmap/, tests/, scripts/ and the top-level
#     docs, and it names whatever it left behind rather than quietly
#     skipping it
#   - commit a file that looks like a secret, a database or market data,
#     whatever .gitignore happens to say today
#   - push a tree whose tests do not pass
#
# Every one of those is something that has already gone wrong at least
# once, which is why each is a hard stop rather than a warning.

set -euo pipefail

PUSH=1
DRY=0
ZIP=""
MSG=""

while [ $# -gt 0 ]; do
    case "$1" in
        --no-push) PUSH=0 ;;
        --dry-run) DRY=1; PUSH=0 ;;
        -h|--help)
            awk 'NR==1{next} /^#/{sub(/^# ?/,""); print; next} {exit}' "$0"
            exit 0 ;;
        -*) echo "unknown option: $1" >&2; exit 2 ;;
        # The zip is recognised by being a zip rather than by its position,
        # so `ship.sh "message"` with no delivery works, and the two cannot
        # be swapped by accident.
        *.zip) ZIP="$1" ;;
        *) if [ -z "$MSG" ]; then MSG="$1"
           else die "did not understand '$1' -- a .zip and a message, in any order"
           fi ;;
    esac
    shift
done

say()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
die()  { printf '\n\033[31mstopped: %s\033[0m\n' "$*" >&2; exit 1; }

# ---- where am I -----------------------------------------------------------
#
# Unzipping a delivery in the wrong directory scatters a liqmap/ tree
# wherever you happened to be standing, and the mess looks like a working
# repo until the push goes to the wrong place.

[ -d .git ] || die "no .git here -- run this from the repo root (cd ~/Downloads/chrischart)"
[ -f liqmap/web.py ] || die "this does not look like the chrischart repo: no liqmap/web.py"

# ---- unpack ---------------------------------------------------------------

if [ -n "$ZIP" ]; then
    [ -f "$ZIP" ] || die "no such zip: $ZIP"
    say "unpacking $(basename "$ZIP")"
    unzip -o "$ZIP" | sed 's/^/  /'
else
    say "no zip given -- committing whatever is already in the tree"
fi

# ---- what changed ---------------------------------------------------------

say "what changed"
git status --short

if [ -z "$(git status --porcelain)" ]; then
    echo "  nothing changed. Either the zip was already unpacked, or it"
    echo "  landed somewhere else."
    exit 0
fi

# ---- refuse the things that must never go in ------------------------------
#
# .gitignore is the first line of defence and this is the second. The two
# disagree the moment someone edits one of them, and the expensive
# direction of that disagreement is the one where a key gets committed.

say "checking for things that must not be committed"
BAD=""
while IFS= read -r f; do
    case "$f" in
        .env|.env.*|*/.env|*.db|*.db-*|wallets.json|*/wallets.json|\
        *.dbn|*.dbn.zst|data/*|*/.DS_Store|.DS_Store|*.pem|*.key|\
        *.npy|__pycache__/*|*/__pycache__/*|*.pyc|*.pyo|\
        .pytest_cache/*|*.egg-info/*|.venv/*|venv/*)
            BAD="$BAD  $f
" ;;
    esac
done < <(git status --porcelain | awk '{print $NF}')

if [ -n "$BAD" ]; then
    printf '%s' "$BAD"
    echo
    echo "  Those are all covered by .gitignore in a healthy tree, so if one"
    echo "  is showing up here, .gitignore has lost a line or the file was"
    echo "  committed once already and is now tracked. Fix that rather than"
    echo "  working around it: 'git rm --cached <file>' untracks it without"
    echo "  deleting it."
    die "refusing to commit those"
fi
echo "  clean"

# ---- stage only what belongs ----------------------------------------------
#
# Never `git add .`. A delivery only ever touches these, so anything else
# in the tree is something you were working on, something that escaped
# .gitignore, or something that should not be here at all -- and all three
# deserve to be looked at rather than swept in.

say "staging"
# ONE PATH AT A TIME, AND ONLY IF IT EXISTS. `git add` is atomic across
# pathspecs: name four directories where one is missing and it stages
# NOTHING and exits non-zero. With the error swallowed that reads as a
# successful run that shipped an empty commit, which is the exact failure
# this script is here to prevent.
for f in liqmap tests scripts README.md DEPLOY.md requirements.txt \
         pyproject.toml Procfile railway.json .gitignore ship.sh; do
    [ -e "$f" ] && git add -- "$f"
done

LEFT="$(git status --porcelain | grep -v '^[MARCD]' || true)"
if [ -n "$LEFT" ]; then
    echo
    echo "  left out of this commit on purpose:"
    printf '%s\n' "$LEFT" | sed 's/^/   /'
    echo "  (add them yourself if they belong)"
fi

if [ -z "$(git diff --cached --name-only)" ]; then
    die "nothing staged -- the changes are all in files this script will not add"
fi

git diff --cached --stat | sed 's/^/  /'

# ---- no secrets in the diff ------------------------------------------------
#
# A token was pasted into a terminal once and ended up in a paste in a
# chat. The cheap version of never doing that again is this grep.

say "scanning the staged diff for secrets"
if git diff --cached -U0 \
   | grep -nE '^\+.*(LIQMAP_TOKEN|DATABENTO_API_KEY|AWS_SECRET_ACCESS_KEY|AWS_ACCESS_KEY_ID)[[:space:]]*=[[:space:]]*["'"'"']?[A-Za-z0-9_/+-]{8,}' \
   | grep -v 'os.environ' | grep -v 'getenv'; then
    die "that looks like a real key in the diff -- it belongs in Railway Variables, not in git"
fi
echo "  clean"

# ---- test -----------------------------------------------------------------
#
# The file limit is raised because the suite builds a few hundred app
# instances and macOS defaults to 256 descriptors, which fails as
# OSError: [Errno 24] and looks nothing like what it is.

say "tests"
ulimit -n 4096 2>/dev/null || true
if ! python -m pytest tests/ -q; then
    echo
    echo "  The changes are unpacked and staged, just not committed. Fix"
    echo "  what failed and run this again -- it will pick up from here."
    die "tests failed -- nothing committed, nothing pushed"
fi

if [ "$DRY" = "1" ]; then
    say "dry run -- unpacked and tested, nothing committed"
    exit 0
fi

# ---- commit ---------------------------------------------------------------

if [ -z "$MSG" ]; then
    printf '\ncommit message: '
    read -r MSG
fi
[ -n "$MSG" ] || die "no commit message"

say "committing"
git commit -m "$MSG" | sed 's/^/  /'

if [ "$PUSH" = "0" ]; then
    say "not pushing (--no-push)"
    echo "  when you are ready:  git push"
    exit 0
fi

# ---- push -----------------------------------------------------------------

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
say "pushing $BRANCH"
git push origin "$BRANCH" | sed 's/^/  /'

say "done"
echo "  Railway builds from GitHub, so the deploy starts on its own."
echo "  Watch it:  https://railway.app  ->  your project  ->  Deployments"
