#!/bin/bash

# Borrowed from here: https://github.com/facebook/docusaurus/issues/3475

readme_file='../README.md'
index_file='./docs/index.md'

# README
cat << EOF > $index_file
---
sidebar_position: 0
slug: /
title: Home
---
EOF

cat $readme_file | sed -e 's#<*[/]*html>##g' | sed -e 's#<*[/]*details>##g' >> $index_file

cat << EOF > ./docs/reference/_category_.json
{
  "label": "API Reference",
  "position": 3
}
EOF