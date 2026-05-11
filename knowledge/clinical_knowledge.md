# Clinical Knowledge Base for RAG Retrieval

This document contains clinical knowledge passages that are chunked and indexed
for semantic retrieval by the RAG system. Each section is a self-contained passage.

---

## Glioma — WHO Grading and Molecular Classification

Gliomas are the most common primary malignant brain tumors, arising from glial cells.
Under WHO CNS5 (2021), gliomas are classified by both histological grade and molecular markers.
Grade II (low-grade) gliomas are diffusely infiltrating but slow-growing, while Grade III
(anaplastic) gliomas show increased mitotic activity. Grade IV includes glioblastoma (GBM),
the most aggressive form, with median survival of 14-16 months despite treatment.

Key molecular markers for glioma classification include:
- IDH1/IDH2 mutation status: IDH-mutant gliomas have significantly better prognosis
- 1p/19q codeletion: defines oligodendroglioma lineage, associated with chemosensitivity
- MGMT promoter methylation: predicts response to temozolomide chemotherapy
- TERT promoter mutations: associated with both low-grade and high-grade gliomas
- ATRX loss: characteristic of astrocytic tumors

Gliomas on T1-weighted MRI typically show variable signal intensity depending on grade.
Low-grade gliomas may appear as well-defined areas of low signal, while high-grade gliomas
show irregular enhancement, central necrosis, and perilesional edema on contrast studies.

---

## Meningioma — Clinical Features and Management

Meningiomas arise from arachnoid cap cells of the meninges and account for approximately
37% of all primary central nervous system tumors. The vast majority (80-85%) are WHO Grade I
(benign), with Grade II (atypical, 15-20%) and Grade III (anaplastic/malignant, 1-3%) being
less common.

Characteristic MRI features include: extra-axial location with a broad dural base ("dural tail
sign"), homogeneous contrast enhancement, well-defined margins, and possible hyperostosis of
adjacent bone. On T1-weighted imaging, meningiomas are typically isointense to slightly
hyperintense relative to gray matter.

Management depends on symptoms, size, and growth rate:
- Observation with serial imaging: appropriate for small, asymptomatic meningiomas
- Surgical resection: primary treatment for symptomatic or growing tumors
- Simpson Grade I-V describes extent of surgical resection and recurrence risk
- Stereotactic radiosurgery: option for small tumors or surgical residual
- Recurrence rates: 7-25% for Grade I, 29-52% for Grade II, 50-94% for Grade III

---

## Pituitary Adenoma — Diagnosis and Endocrine Evaluation

Pituitary adenomas are benign neoplasms arising from the anterior pituitary gland, classified
by size as microadenomas (<10mm) or macroadenomas (≥10mm). They account for 10-15% of
intracranial tumors and are found incidentally in up to 20% of autopsies.

Functional classification:
- Prolactinomas (40%): elevated prolactin, amenorrhea/galactorrhea, treated with dopamine agonists
- Non-functioning adenomas (30%): present with mass effect, visual field deficits
- Growth hormone-secreting (20%): acromegaly or gigantism
- ACTH-secreting (5-10%): Cushing's disease
- TSH-secreting (<1%): central hyperthyroidism

MRI findings: sellar/suprasellar mass, variable T1 signal intensity. Macroadenomas may compress
the optic chiasm (bitemporal hemianopia), invade the cavernous sinus, or cause pituitary apoplexy.
Dynamic contrast-enhanced MRI with thin coronal sections is the gold standard for detection.

Endocrine workup should include: prolactin, IGF-1, morning cortisol, ACTH, free T4, TSH,
LH, FSH, testosterone/estradiol. Visual field testing (Humphrey or Goldmann perimetry) is
mandatory for macroadenomas approaching the optic chiasm.

---

## Differential Diagnosis Considerations for Brain Tumors

When evaluating brain lesions on MRI, differential diagnosis should consider:

For intra-axial lesions (within brain parenchyma):
- Primary brain tumors: gliomas (most common), lymphoma, embryonal tumors
- Metastatic disease: lung, breast, melanoma, renal cell carcinoma most common primaries
- Abscess: ring-enhancing with restricted diffusion on DWI
- Demyelinating disease: multiple sclerosis plaques can mimic tumors
- Vascular: cavernous malformation, hemorrhagic infarction

For extra-axial lesions (outside brain parenchyma):
- Meningioma: most common extra-axial tumor
- Schwannoma: typically in cerebellopontine angle
- Epidermoid/dermoid cysts
- Metastatic dural disease

Key discriminating features include: enhancement pattern, diffusion restriction, perfusion
characteristics, spectroscopy findings, and clinical presentation including patient age
and symptom onset.

---

## Imaging Protocols for Brain Tumor Assessment

Standard brain tumor MRI protocol should include:

Core sequences:
- T1-weighted: anatomical detail, baseline for contrast comparison
- T2-weighted: edema detection, lesion characterization
- FLAIR (Fluid-Attenuated Inversion Recovery): perilesional edema, infiltrative disease
- T1-weighted post-gadolinium: blood-brain barrier breakdown, tumor enhancement
- DWI/ADC: cellularity assessment, abscess vs tumor differentiation

Advanced sequences for tumor grading:
- Dynamic susceptibility contrast (DSC) perfusion: relative cerebral blood volume (rCBV)
  - High rCBV suggests high-grade tumor
  - Low rCBV more consistent with low-grade or non-neoplastic lesion
- MR spectroscopy: metabolite ratios (Cho/NAA, Cho/Cr)
  - Elevated choline: increased membrane turnover (tumor)
  - Decreased NAA: neuronal loss
  - Lactate peak: anaerobic metabolism (high-grade)
- Diffusion tensor imaging (DTI): white matter tract involvement

Measurement standards for tumor assessment:
- RANO criteria: Response Assessment in Neuro-Oncology
- Bidimensional measurements of enhancing disease
- T2/FLAIR changes assessed qualitatively
- Clinical assessment and corticosteroid dose must be documented

---

## Treatment Pathways by Tumor Type

Glioma treatment pathways (per NCCN Guidelines):
- Grade II: maximal safe resection → observation vs. radiation + chemotherapy
- Grade III: maximal safe resection → radiation + chemotherapy (PCV or temozolomide)
- Grade IV (GBM): maximal safe resection → concurrent chemoradiation (Stupp protocol) →
  adjuvant temozolomide × 6-12 cycles
- Tumor Treating Fields (TTFields/Optune): FDA-approved for newly diagnosed and recurrent GBM
- Bevacizumab: FDA-approved for recurrent GBM

Meningioma treatment pathways:
- Grade I: observation if asymptomatic; surgical resection if symptomatic
- Grade II: surgical resection ± adjuvant radiation
- Grade III: surgical resection + adjuvant radiation

Pituitary adenoma treatment pathways:
- Prolactinomas: dopamine agonists (cabergoline, bromocriptine) as first-line
- Non-functioning: transsphenoidal surgery if symptomatic or growing
- GH-secreting: transsphenoidal surgery; somatostatin analogues if surgery insufficient
- ACTH-secreting: transsphenoidal surgery; medical therapy (ketoconazole, pasireotide) if needed

---

## Safety Considerations for AI-Assisted Radiology

Key safety principles for AI in medical imaging:

Regulatory framework:
- FDA classifies AI-based radiology tools as Software as a Medical Device (SaMD)
- EU AI Act classifies medical diagnostic AI as HIGH-RISK (Annex III)
- Article 14: mandatory human oversight for high-risk AI systems
- Article 9: risk management system required throughout AI lifecycle

Clinical deployment considerations:
- AI should augment, not replace, radiologist judgment
- All AI outputs must be reviewed by a qualified healthcare professional
- Confidence scores must be provided and explained to clinical users
- Known limitations and failure modes must be documented
- Regular performance monitoring against clinical ground truth is required
- Bias assessment across patient demographics (age, sex, ethnicity) is mandatory

Quality assurance for AI radiology systems:
- Input quality validation: reject artifacts, motion-degraded, or protocol-variant images
- Calibration: ensure confidence scores are well-calibrated (not overconfident)
- Hallucination detection: verify findings are supported by image evidence
- Consistency: same input should produce consistent outputs
- Edge case documentation: known failure modes and distribution shifts
