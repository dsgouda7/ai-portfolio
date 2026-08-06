# Setup

The chapter is offline and standard-library only. Setup creates a virtual
environment, installs Jupyter support, registers a chapter-local kernel, and
updates notebook kernel metadata. It does not authenticate to Azure, install an
Azure SDK, read environment secrets, or contact customer systems.

## Requirements

- Python 3.10 or newer
- PowerShell on Windows, or a POSIX shell on macOS/Linux
- Network access only if the listed Jupyter packages are not already cached

## Windows

```powershell
cd learning\fde\04-identity-isolation-and-compliance
.\setup.ps1
```

Use `-SkipKernel` to install packages without registering or assigning a kernel.

## macOS or Linux

```bash
cd learning/fde/04-identity-isolation-and-compliance
./setup.sh
```

Use `--skip-kernel` for dependency installation only.

## Safety boundary

- Do not add `.env` files, tokens, client secrets, certificates, or connection strings.
- Do not replace synthetic IDs with customer or production identifiers.
- Do not run cloud smoke tests from this notebook.
- Store later environment evidence in an approved evidence system, not notebook outputs.

The implementation task that created this chapter did not run either setup script
and did not execute the notebook.
