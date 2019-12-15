#!/bin/bash bash

# Install kcov on Travis CI, run the executables to collect code coverage statistics, and send them to Codecov.

if [[ "${TRAVIS_OS_NAME}" == "linux" ]]
then
    # Install kcov.
    wget https://github.com/SimonKagstrom/kcov/archive/master.tar.gz &&
    tar xzf master.tar.gz &&
    cd kcov-master &&
    mkdir build &&
    cd build &&
    cmake .. &&
    make &&
    sudo make install &&
    cd ../.. &&
    rm -rf kcov-master &&
    ls -lah target/debug/* &&

    # Collect the coverage statistics.
    for file in target/debug/deps/*[^\.d]
    do
        mkdir -p "target/cov/$(basename ${file})"
        kcov --exclude-pattern=/.cargo,/usr/lib,tests --exclude-region='#[cfg(test)]:#[cfg(testkcovstopmarker)]' --verify "target/cov/$(basename ${file})" "${file}"
    done &&

    # Upload the coverage report to Codecov.
    bash <(curl -s https://codecov.io/bash) &&
    echo "Uploaded code coverage"
fi
