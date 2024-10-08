---
name: Raise an Issue
about: Describe this issue template's purpose here.
title: Difficulty Level
labels: ''
assignees: ''

---

name: "Raise an Issue"
description: "Select the difficulty level for your issue"
title: "[Issue]:"
labels: enhancement
body:
  - type: dropdown
    id: difficulty
    attributes:
      label: "What is the difficulty level of this project for you?"
      description: "Please select the difficulty level"
      options:
        - Level 1 (Beginner)
        - Level 2 (Intermediate)
        - Level 3 (Advanced)
      required: true
