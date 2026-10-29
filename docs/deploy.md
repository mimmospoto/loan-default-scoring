# Deployment

The service runs on Azure Container Apps in North Europe. Every push to `main`
that passes the tests is deployed automatically.

## How a change goes live

1. A push to `main` starts the workflow in
   [`.github/workflows/ci.yml`](../.github/workflows/ci.yml).
2. **test** builds the image and runs all tests, including the container check.
3. **publish** pushes the image to `ghcr.io/mimmospoto/loan-default-scoring`,
   tagged `latest` and with the short commit hash.
4. **deploy** logs in to Azure, points the container app at the new tag, and
   calls `/health` until the service answers. If it never does, the run fails.

Pull requests run only **test**.

## What runs in Azure

| Piece | Value |
|---|---|
| Region | North Europe |
| Resource group | `rg-loan-default` |
| Container app | `loan-default-api` (environment `cae-loan-default`) |
| Image | public, from GitHub Container Registry |
| Size | 0.5 vCPU, 1 GiB |
| Replicas | minimum 0, maximum 1 |
| Ingress | external HTTPS, port 8000 |
| Secret | `api-key`, exposed to the app as `API_KEY` |

## Design decisions

- **Container Apps, not Kubernetes.** The service is one stateless container
  with one model. Container Apps already gives HTTPS, a public address,
  restarts, revisions and scaling to zero. Kubernetes would add a cluster to
  operate (nodes, upgrades, ingress, certificates) and a cost floor for nothing
  this service needs. It would make sense with many services, custom
  networking, or a platform team.
- **Scale to zero.** With no traffic the app has no replicas and is not
  billed. The first request after a pause waits for a cold start (10 to 30
  seconds in testing). That suits a demo; a real lender would set a minimum of
  one replica.
- **Maximum of one replica.** It caps the cost and the load. One replica
  handles this model comfortably, since scoring is a few milliseconds.
- **Passwordless login.** The deploy job proves its identity to Azure with
  OIDC. GitHub holds three identifiers (client, tenant, subscription), not a
  password. The trust only accepts the `main` branch of this repo, and the
  identity has Contributor on the one resource group, nothing wider.
- **A new tag per deploy.** The app is pointed at the commit hash, not
  `latest`. A tag that never changes would not create a new revision, and the
  running version could not be told from the repository.
- **The key lives in an Azure secret,** not in the image or the repo. The
  service refuses scoring calls without the matching `X-API-Key` header. The
  key is shared privately with people who want to try the service.
- **A public image.** It contains only code and a 371 KB model, no data or
  secrets, so Azure can pull it without credentials.

## Cost

Azure's free grant for Container Apps (Consumption plan) is the first 180,000
vCPU-seconds, 360,000 GiB-seconds and 2 million requests per subscription per
month. At 0.5 vCPU that is roughly 100 active hours. With scale to zero, a demo
should stay inside it.

Not verified: the per-second prices outside the grant. Azure's pricing page
showed placeholders when checked, so no dollar figure is given here. A budget
alert on the subscription warns before spending gets large, and Cost analysis
in the portal shows the real cost per resource.

## Keeping the address private

The address is not in the repo. Any request, even one rejected with a 401,
wakes the app and counts as active time, so a bot hitting a public address
could keep it running all month. The address and the key are shared together
with people who ask. If the cost ever grows, deactivate the revision.

## Operations

- **Check the service:** `GET /health` returns the model version that is loaded.
- **See what is running:** the container app's *Revisions and replicas* page
  shows the active revision, named after the deployed commit.
- **Roll back:** run `az containerapp update` with an older image tag, or
  re-run the workflow on an older commit.
- **Rotate the key:** change the `api-key` secret in Azure, then start a new
  revision so the app picks it up.

## Limitations

- One shared key, no per-caller keys, no rate limit.
- A cold start after idle time.
- The publish job rebuilds the image rather than reusing the tested one.
- No staging environment: a passing build goes straight to the live service.
- Scored requests are logged to stdout and so reach Azure's logs, but nothing reads them on a schedule; see [monitoring.md](monitoring.md).
