#!/bin/bash bash

# Create the library documentation and publish it on GitHub pages.

if [[ "${TRAVIS_RUST_VERSION}" == "stable" && "${TRAVIS_OS_NAME}" == "linux" ]]
then
  cargo rustdoc --lib --all-features -- --document-private-items &&
  echo "<meta http-equiv=refresh content=0;url=${PROJECT_NAME}/index.html>" > target/doc/index.html &&
  sudo pip install ghp-import &&
  ghp-import -n target/doc &&

  # Upload the documentation to GitHub.
  git push -qf https://${GITHUB_TOKEN}@github.com/${TRAVIS_REPO_SLUG}.git documentation;
  echo "Uploaded documentation"
fi;
