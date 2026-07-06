# Security Policy

## Reporting a vulnerability

LaneVision is a computer-vision demo, not a production service, but if you find
a security issue (for example in the FastAPI server or file-upload handling),
please report it privately rather than opening a public issue.

Open a [GitHub security advisory][advisory] or contact the maintainer directly.
Please include steps to reproduce and, if possible, a suggested fix. We aim to
acknowledge reports within a few days.

## Scope notes

- The upload endpoint accepts video files and writes them to a local `uploads/`
  directory. Run the server only on trusted networks; it is not hardened for
  public internet exposure.
- The webcam and video sources are processed entirely locally; no data leaves
  the machine.

[advisory]: https://github.com/rishii-hub/realtime-lane-detection/security/advisories/new
