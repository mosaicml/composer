#!/bin/bash

# Borrowed from here: https://github.com/facebook/docusaurus/issues/3475

readme_file='../README.md'
index_file='./docs/index.md'

# README
cat << EOF > $index_file
---
slug: /
title: Home
---
EOF

#tail -n +3 $readme_file >> $index_file
# Cat and Strip out html and details tags
cat $readme_file | sed -e 's#<*[/]*html>##g' | sed -e 's#<*[/]*details>##g' >> $index_file


contrib_file='../CONTRIBUTING.md'
to_contrib_file='./docs/contributing.md'

# Contributing
cat $contrib_file > $to_contrib_file
