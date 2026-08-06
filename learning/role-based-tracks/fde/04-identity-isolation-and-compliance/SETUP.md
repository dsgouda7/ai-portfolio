# Set Up the Local Identity Lab

This setup creates a local Jupyter environment for the synthetic Riverside
identity cases. It does not sign in to Azure, read secrets, call an identity
provider, or connect to customer systems.

## Before you start

You need:

- Python 3.10 or newer;
- PowerShell on Windows, or a POSIX shell on macOS/Linux;
- network access only when the Jupyter packages are not already cached.

The safe starting state is simple: use only the committed fixture IDs. Do not
add `.env` files, tokens, client secrets, certificates, connection strings,
production exports, or customer identifiers.

## Windows

From the repository root:

```powershell
cd learning\role-based-tracks\fde\04-identity-isolation-and-compliance
.\setup.ps1
```

To install packages without registering or assigning the notebook kernel:

```powershell
.\setup.ps1 -SkipKernel
```

## macOS or Linux

From the repository root:

```bash
cd learning/role-based-tracks/fde/04-identity-isolation-and-compliance
./setup.sh
```

To install packages without registering or assigning the notebook kernel:

```bash
./setup.sh --skip-kernel
```

## Confirm the boundary before running

Ask these questions before opening the notebook:

1. Are all identities and resources from `fixtures/identity-scenarios-v1.json`?
2. Are there no secrets or live endpoints in the environment?
3. Will any completed report be stored separately from the committed `NOT RUN` example?

If any answer is no or unknown, stop. This lab is not a cloud smoke test.

## What setup proves

A successful setup proves only that the local environment and chapter kernel can
be created. After you run the notebook and record the result, you may have
`[Local-measured]` evidence for the synthetic cases. Neither setup nor a green
notebook proves deployed RBAC, identity revocation, residency, privacy,
compliance, or legal approval.

Store later environment evidence in an approved evidence system, not in notebook
outputs. The chapter's original implementation did not run these setup scripts
or leave executed outputs in the committed notebook.
