# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.x     | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take security seriously. If you discover a vulnerability, please **do not**
open a public issue.

Instead, email **rishii2526@gmail.com** with:

- A description of the vulnerability and its potential impact
- Steps to reproduce
- Any suggested remediation

You can expect an acknowledgement within **72 hours** and a status update within
**7 days**. Once the issue is resolved we will credit you (unless you prefer to
remain anonymous) in the release notes.

## Scope

This is a client-side computer-vision project with no backend, authentication,
or network services. The most relevant concerns are:

- Malicious media files that could trigger decoder vulnerabilities in OpenCV
- Supply-chain integrity of Python and npm dependencies

Please keep dependencies up to date and report any suspicious behavior.
