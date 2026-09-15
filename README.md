RICD Higher-Education Collapse Tracker

A Bayesian state-space model and validated classifier for institutional financial collapse risk in U.S. higher education, built on the Recursive Information-Container Dynamics (RICD) framework.

The full RICD manuscript (the complete, domain-independent theory) lives at docs/RICD_15_6_master.docx (also available as .tex). Everything in this repository's code implements a real subset of that framework (Parts 6, 7, 8, and 10.5b specifically) against U.S. higher-education data; the manuscript itself is domain-independent and covers considerably more than the tracker uses. The manuscript's own Integration Manifest records every mechanism confirmed built into the framework, section by section, with the exact manuscript text shown for each -- a standing verification record kept specifically to be checked against the manuscript, not trusted on its own.

The original source documents RICD was built from live in source-documents/: the quartet of poems, the Pentagonal Theorem of the Mathematical Nature of Evil, and Shaking Bowls — Carmen Speer's own original creative and theoretical writing, which became FDFM and RICS respectively, then nested into RICS-FDFM, then RICD, before further revisions arrived at the manuscript above. The folder's intermediate-development/ subfolder holds real, surviving milestones from that path: an early FDFM application proposing a justice-system tracker (directly cited by RICD 5.0's own editorial notes, not a lost document), the expanded RICS-FDFM formalization, and earlier RICD versions (5.0 through 5.6). See that folder's own README for the full lineage.

The narrative, findings, and process record of how this tracker was built live in reports/: RICD_Tracker_Findings Final.pdf, RICD_Tracker_Narrative final.pdf, and RICD_Tracker_Process_Log Final.pdf -- the findings document, the narrative account of how each result was actually reached, and the consolidated process log -- alongside Higher_Ed_Sector_Findings.docx (a separate analysis of what the results imply about the U.S. higher-education sector as a whole), Claude's Account of Carmen's Role in Building RICD and the higher-ed tracker.pdf and ChatGPT's Account of Its own Role in Building FDFM, RICS, and RICD 1.0.pdf (each AI collaborator's own account of working with Carmen on the framework and the tracker, asked for and included so the actual division of labor is checkable rather than asserted), RICD_Adapter_Instructional_Manual_3.pdf (an instructional manual for engineers working with the RICD adapter contract directly), Actor_Tracker_Seed_Note.docx (a seed note for a genuinely different kind of tracker planned for later), and the complete RICD_Integration_Manifest_Complete.pdf described above. An interlinear Source Translation Ledger (source-documents/Source_Translation_Ledger_ RICD source poems explained mathematically.pdf) is also included, setting RICD's own mathematics directly alongside the four original poems it was built from, line by line, so the connection between the source material and the formal framework is checkable rather than asserted.

The planned future tracker mentioned above is an actor tracker, which would investigate the specific real decisions and actors behind a collapse (board minutes, depositions, investigative findings), rather than the aggregate financial and enrollment effects this higher-ed tracker measures. The two are complementary, not competing — see the seed note itself for why an actor tracker needs a different kind of evidence entirely, and why it wasn't attempted here.

What this is

A real, working pipeline — not a report about known outcomes — that:

Models an institution's official and operational representations as latent stochastic processes (src/model.py), including a genuine shock-type latent process for debt (src/jump_diffusion.py), since debt behaves as long stable stretches punctuated by rare large jumps rather than continuous drift.
Extracts eight validated, independently-tested features from real IPEDS data: informational trajectory (d_A trend and endpoint), resource debt severity and its own trend (δR, δR-trend), a debt-spike signature, a regime-classification fraction, scale-normalized reserve adequacy (endowment relative to enrollment, not to debt — see the note on that below), and a research-to-instruction expenditure ratio.
Applies a direct External Governance Attestation override for institutions with a real, independent, confirmed governance verdict (an accreditor's show-cause order or withdrawal), bypassing the statistical classifier entirely for those cases rather than fitting it as a weighted feature — confirmed empirically to be the correct approach (src/classifier.py).
Detects shared external shocks across the panel from cross-sectional co-movement alone (src/common_cause.py), run on raw adapter-level data rather than model output, since latent-variable smoothing washes out exactly the sharp signal this diagnostic needs.
Validated result

100.00% leave-one-out cross-validated accuracy on a real, 54-institution panel (23 confirmed closures spanning seven distinct collapse mechanisms, 31 confirmed-stable comparisons), with zero misclassifications. Reproduce this directly:

bash
pip install -r requirements.txt
cd src
python -c "
from classifier import RICDClassifier, load_panel
panel = load_panel('../data/panel/panel.json')
clf = RICDClassifier()
acc, misclassified = clf.leave_one_out_accuracy(panel)
print(f'Accuracy: {acc:.2%}')
print(f'Misclassified: {misclassified}')
"

A live public dashboard of the validated panel is at carmen-speer.github.io/Recursive-Information-Container-Dynamics (built from docs/index.html and docs/data/panel.json via GitHub Pages).

The panel is a validation set, not a survey

The 54 institutions in data/panel/panel.json were assembled specifically to test the classifier, including a deliberate rebalancing pass mid-project after an audit found the panel had drifted toward roughly three times the real-world rate of dramatic, easily-searchable closure cases. Nothing here should be read as a claim about what fraction of U.S. higher education is at risk. It supports a narrower, real claim: these specific mechanisms and relationships showed up clearly enough in independently-verified data to resolve a hard classification problem.

Known gaps — stated honestly, not smoothed over

This is a live, ongoing project, not a finished product, and it's more useful to state clearly what still needs real work than to imply everything below is complete:

The live-scoring pipeline (src/score_institution.py) is fully wired end to end, but not yet confirmed working against live data, and it does not yet write its output anywhere for the scheduled workflow to publish — right now it only prints one institution's classification to the workflow log as a pipeline smoke test. There is no results/ directory in this repository yet; .github/workflows/rescore.yml's "commit updated results" step currently has nothing to commit. Turning this into an actual self-updating tracker means writing a script that scores more than one institution and saves the output somewhere the dashboard can read — genuine, not-yet-done work, not a configuration problem.
IPEDS finance data has no stable API, and its distribution mechanism has already changed more than once during this project. The original bulk-file URL pattern (nces.ed.gov/ipeds/datacenter/data/F....zip) is defunct; src/fetch_live_data.py has been updated to the current, real endpoint (nces.ed.gov/ipeds/data-generator?..., confirmed directly against NCES's own live Complete Data Files page), which also returns a raw CSV rather than a zip archive. This part of the pipeline has never been exercised end-to-end from any sandboxed AI environment used on this project, since nces.ed.gov and api.data.gov are outside every one of those sandboxes' network allowlists — confirmed directly, repeatedly. GitHub Actions' own runners are not under that restriction, so a manual or scheduled run of rescore.yml is the actual first real test of whether this works, not a repeat of an already-known result. NCES has changed this endpoint before and may again — automated re-scoring should be monitored, not trusted blindly, on this specific point.
Four institutions in the original research (three small closed colleges, one for-profit) have no real endowment data to find — three because it was only located after checking earlier filing years than initially tried, one (a for-profit) because for-profit institutions do not report an endowment field at all, a genuine structural fact rather than a gap.
The harder half of the coupling machinery (asymmetric, predatory resource extraction between containers, as opposed to a clean merger or a mutual-benefit arrangement) has one real, worked instance (documented in the manuscript) but has not been validated against a full retrospective panel fit — the subordinate institution's own chaotic collapse left no single clean container to test against.
Repository structure
src/
  model.py               Bayesian state-space model (PyMC)
  jump_diffusion.py       Shock-type latent process for debt
  common_cause.py         Shared-external-shock detector
  real_adapter.py          Real IPEDS/Scorecard data loading (historical, local files)
  fetch_live_data.py       Live data fetching (College Scorecard API + IPEDS bulk files)
  classifier.py             The 8-feature + governance-override classifier
  score_institution.py       End-to-end scoring entry point (see Known Gaps)
  dynamics.py                 Core RICD dynamical-system equations used by the model
  legacy_peer_density_reference.py   Retained reference implementation from an earlier peer-density approach
data/
  panel/panel.json           The real, validated 54-institution panel
docs/
  RICD_15_6_master.docx, .tex    The full, domain-independent RICD theory
  index.html                      Public results dashboard (GitHub Pages)
  data/panel.json                  Panel data the dashboard reads from
source-documents/
  Quartet_of_poems.pdf                                                 Original poems
  The_Pentagonal_Theorem_of_the_Mathematical_Nature_of_Evil_-2.pdf       Became FDFM
  Shaking_Bowls_Thought_Experiment-1.pdf                                 Became RICS
  Source_Translation_Ledger_ RICD source poems explained mathematically.pdf   Poems set line-by-line alongside RICD's math
  README.md                                                               Full lineage
  intermediate-development/
    Feedback_Divergence_Field_Model...justice_system....docx               Early FDFM justice-tracker proposal
    RICS_FDFM_Multiscale_Information_Geometric_Model.pdf                    Expanded nested RICS-FDFM
    RICD_5_0.docx, RICD_5_3.pdf, RICD_5_4.pdf,                              Earlier RICD versions
    RICD_5_5.docx, RICD_5_6.pdf, RICD_1_2_or_1_3_early_version.pdf
reports/
  RICD_Tracker_Findings Final.pdf        Final findings document
  RICD_Tracker_Narrative final.pdf       Narrative account of how results were reached
  RICD_Tracker_Process_Log Final.pdf     Consolidated process record
  Higher_Ed_Sector_Findings.docx          What the results imply about the sector
  Claude's Account of Carmen's Role in Building RICD and the higher-ed tracker.pdf   Claude's own account of the collaboration
  ChatGPT's Account of Its own Role in Building FDFM, RICS, and RICD 1.0.pdf          ChatGPT's own account of the collaboration
  Actor_Tracker_Seed_Note.docx             Seed note for a mechanism-layer (actor) tracker, planned for later
  RICD_Adapter_Instructional_Manual_3.pdf  Adapter-contract implementation guide, for engineers
  RICD_Integration_Manifest_Complete.pdf   Every mechanism confirmed built into RICD, with the manuscript text shown for each
.github/workflows/
  rescore.yml                 Scheduled re-scoring workflow
Re-scoring cadence

IPEDS is not live data — it releases on a fixed institutional schedule (provisional data a few times a year, final data annually). The scheduled workflow in .github/workflows/rescore.yml runs periodically and checks for new data rather than assuming a fixed release date; a run that finds nothing new is a normal, expected outcome, not a failure.
