## default track, shared view, privacy-synthetic

| entry | mR | mP | mF2 | doc leak | pF1 | mF2 out-of-domain (n sets) |
|---|---|---|---|---|---|---|
| bardsai-v3 | 0.976 | 0.825 | 0.921 | 0.047 | 0.905 | 0.842 (3) |
| opf | 0.915 | 0.771 | 0.863 | 0.123 | 0.798 | n/a (0) |
| sct-default | 0.949 | 0.613 | 0.794 | 0.171 | 0.453 | 0.920 (2) |
| gliner2-fastino | 0.920 | 0.623 | 0.788 | 0.134 | 0.854 | 0.788 (7) |
| gliner-nvidia | 0.847 | 0.657 | 0.765 | 0.212 | 0.832 | 0.651 (3) |
| presidio | 0.861 | 0.636 | 0.758 | 0.279 | 0.700 | 0.758 (7) |
| sct-gliner | 0.952 | 0.479 | 0.734 | 0.163 | 0.447 | n/a (0) |
| datafog | 0.678 | 0.828 | 0.683 | 0.507 | 0.659 | 0.683 (7) |

## default track, shared view, privacy-real

| entry | mR | mP | mF2 | doc leak | pF1 | mF2 out-of-domain (n sets) |
|---|---|---|---|---|---|---|
| datafog | 0.768 | 0.750 | 0.750 | 0.812 | 0.819 | 0.750 (2) |
| sct-gliner | 0.980 | 0.206 | 0.554 | 0.182 | 0.295 | n/a (0) |
| sct-default | 0.982 | 0.195 | 0.539 | 0.181 | 0.298 | n/a (0) |
| gliner-nvidia | 0.401 | 0.478 | 0.388 | 0.513 | 0.537 | 0.388 (2) |
| opf | 0.469 | 0.426 | 0.372 | 0.622 | 0.434 | n/a (0) |
| gliner2-fastino | 0.487 | 0.181 | 0.354 | 0.615 | 0.462 | 0.354 (2) |
| bardsai-v3 | 0.455 | 0.193 | 0.351 | 0.646 | 0.450 | 0.351 (2) |
| presidio | 0.456 | 0.172 | 0.323 | 0.721 | 0.363 | 0.323 (2) |

## default track, shared view, ner

| entry | mR | mP | mF2 | doc leak | pF1 | mF2 out-of-domain (n sets) |
|---|---|---|---|---|---|---|
| bardsai-v3 | 0.500 | 0.100 | 0.119 | 0.500 | 0.500 | 0.119 (5) |
| gliner2-fastino | 0.833 | 0.040 | 0.102 | 0.250 | 0.870 | 0.102 (5) |
| opf | 0.583 | 0.044 | 0.099 | 0.500 | 0.600 | n/a (0) |
| sct-default | 1.000 | 0.026 | 0.086 | 0.000 | 0.015 | n/a (0) |
| presidio | 0.583 | 0.040 | 0.084 | 0.625 | 0.522 | 0.084 (5) |
| sct-gliner | 1.000 | 0.025 | 0.083 | 0.000 | 0.030 | n/a (0) |
| gliner-nvidia | 0.417 | 0.038 | 0.068 | 0.625 | 0.316 | 0.068 (5) |
| datafog | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 (5) |

## default track, full view, privacy-synthetic

| entry | mR | mP | mF2 | doc leak | pF1 | mF2 out-of-domain (n sets) |
|---|---|---|---|---|---|---|
| bardsai-v3 | 0.908 | 0.917 | 0.910 | 0.214 | 0.858 | 0.814 (3) |
| sct-default | 0.926 | 0.757 | 0.877 | 0.367 | 0.598 | 0.949 (2) |
| gliner2-fastino | 0.883 | 0.795 | 0.860 | 0.382 | 0.795 | 0.860 (7) |
| sct-gliner | 0.863 | 0.667 | 0.798 | 0.483 | 0.470 | n/a (0) |
| opf | 0.755 | 0.880 | 0.773 | 0.462 | 0.692 | n/a (0) |
| gliner-nvidia | 0.770 | 0.823 | 0.772 | 0.493 | 0.733 | 0.625 (3) |
| presidio | 0.705 | 0.730 | 0.697 | 0.645 | 0.557 | 0.697 (7) |
| datafog | 0.337 | 0.854 | 0.379 | 0.919 | 0.377 | 0.379 (7) |

## default track, full view, privacy-real

| entry | mR | mP | mF2 | doc leak | pF1 | mF2 out-of-domain (n sets) |
|---|---|---|---|---|---|---|
| sct-default | 0.932 | 0.525 | 0.804 | 0.828 | 0.564 | n/a (0) |
| gliner2-fastino | 0.868 | 0.598 | 0.793 | 0.849 | 0.669 | 0.793 (2) |
| bardsai-v3 | 0.720 | 0.809 | 0.719 | 0.833 | 0.671 | 0.719 (2) |
| sct-gliner | 0.678 | 0.485 | 0.618 | 0.923 | 0.380 | n/a (0) |
| presidio | 0.532 | 0.631 | 0.517 | 0.951 | 0.440 | 0.517 (2) |
| opf | 0.460 | 0.819 | 0.468 | 0.903 | 0.522 | n/a (0) |
| gliner-nvidia | 0.403 | 0.807 | 0.435 | 0.911 | 0.495 | 0.435 (2) |
| datafog | 0.161 | 0.765 | 0.183 | 0.997 | 0.235 | 0.183 (2) |

## default track, full view, ner

| entry | mR | mP | mF2 | doc leak | pF1 | mF2 out-of-domain (n sets) |
|---|---|---|---|---|---|---|
| sct-default | 0.848 | 0.693 | 0.807 | 0.363 | 0.784 | n/a (0) |
| gliner2-fastino | 0.739 | 0.752 | 0.726 | 0.391 | 0.636 | 0.726 (5) |
| gliner-nvidia | 0.287 | 0.656 | 0.305 | 0.795 | 0.246 | 0.305 (5) |
| bardsai-v3 | 0.256 | 0.921 | 0.293 | 0.872 | 0.302 | 0.293 (5) |
| sct-gliner | 0.271 | 0.353 | 0.276 | 0.858 | 0.166 | n/a (0) |
| opf | 0.157 | 0.815 | 0.183 | 0.909 | 0.155 | n/a (0) |
| presidio | 0.095 | 0.469 | 0.105 | 0.962 | 0.045 | 0.105 (5) |
| datafog | 0.001 | 0.333 | 0.001 | 1.000 | 0.000 | 0.001 (5) |

## default track, per dataset, full view, masked F2 (completed docs)

| dataset | bardsai-v3 | datafog | gliner-nvidia | gliner2-fastino | opf | presidio | sct-default | sct-gliner |
|---|---|---|---|---|---|---|---|---|
| openpii | 0.969 | 0.338 | 0.836 | 0.854 | 0.839 | 0.647 | 0.905 | 0.859 |
| gretel | 0.660 | 0.150 | 0.713 | 0.690 | 0.540 | 0.472 | 0.576 | 0.562 |
| gretel_en | 0.994 | 0.487 | 0.933 | 0.908 | 0.903 | 0.865 | 0.933 | 0.929 |
| nemotron | 0.979 | 0.357 | 0.867 | 0.879 | 0.649 | 0.780 | 0.870 | 0.844 |
| argilla_pii | 0.983 | 0.233 | 0.895 | 0.856 | 0.747 | 0.880 | 0.964 | 0.935 |
| thai_pii | 0.889 | 0.561 | 0.439 | 0.930 | 0.844 | 0.486 | 0.967 | 0.687 |
| bayanat | 0.893 | 0.527 | 0.722 | 0.914 | 0.887 | 0.751 | 0.927 | 0.772 |
| tab | 0.638 | 0.093 | 0.308 | 0.783 | 0.229 | 0.486 | 0.831 | 0.558 |
| redact | 0.801 | 0.274 | 0.562 | 0.802 | 0.708 | 0.549 | 0.777 | 0.677 |
| iahlt | 0.433 | 0.000 | 0.598 | 0.864 | 0.118 | 0.188 | 0.899 | 0.447 |
| thainer | 0.394 | 0.004 | 0.077 | 0.568 | 0.325 | 0.107 | 0.860 | 0.139 |
| naamapadam | 0.391 | 0.000 | 0.030 | 0.883 | 0.233 | 0.102 | 0.707 | 0.051 |
| multiconer | 0.225 | 0.000 | 0.211 | 0.628 | 0.185 | 0.076 | 0.740 | 0.194 |
| stockmark | 0.024 | 0.000 | 0.608 | 0.687 | 0.056 | 0.051 | 0.832 | 0.550 |

## default track, completion

| entry | docs | failed | flagged datasets (< 99% complete) |
|---|---|---|---|
| bardsai-v3 | 16661 | 0 | - |
| datafog | 16661 | 0 | - |
| gliner-nvidia | 16661 | 0 | - |
| gliner2-fastino | 16661 | 9 | - |
| opf | 16661 | 0 | - |
| presidio | 16661 | 0 | - |
| sct-default | 16661 | 0 | - |
| sct-gliner | 16661 | 0 | - |

## default track, paired bootstrap vs baseline

| entry | baseline better | entry better | no difference (masked_recall, shared view) |
|---|---|---|---|
| bardsai-v3 | bayanat, redact, tab, thainer | gretel, gretel_en, nemotron, openpii | 6 |
| datafog | argilla_pii, bayanat, gretel, gretel_en, nemotron, openpii, redact, tab, thai_pii, thainer | - | 4 |
| gliner-nvidia | argilla_pii, bayanat, nemotron, openpii, redact, tab, thai_pii, thainer | gretel | 5 |
| gliner2-fastino | argilla_pii, bayanat, openpii, redact, tab, thai_pii | gretel | 7 |
| opf | argilla_pii, bayanat, nemotron, redact, tab, thai_pii | openpii | 7 |
| presidio | argilla_pii, bayanat, gretel_en, nemotron, openpii, redact, tab, thai_pii, thainer | - | 5 |
| sct-gliner | redact | openpii | 12 |

## tuned track, shared view, privacy-synthetic

| entry | mR | mP | mF2 | doc leak | pF1 | mF2 out-of-domain (n sets) |
|---|---|---|---|---|---|---|
| bardsai-v3 | 0.976 | 0.825 | 0.921 | 0.047 | 0.905 | 0.842 (3) |
| opf | 0.944 | 0.720 | 0.858 | 0.086 | 0.787 | n/a (0) |
| gliner-nvidia | 0.938 | 0.633 | 0.819 | 0.118 | 0.830 | 0.667 (3) |
| sct-default | 0.949 | 0.611 | 0.794 | 0.171 | 0.453 | 0.919 (2) |
| gliner2-fastino | 0.944 | 0.573 | 0.778 | 0.097 | 0.846 | 0.778 (7) |
| presidio | 0.861 | 0.636 | 0.758 | 0.279 | 0.700 | 0.758 (7) |
| sct-gliner | 0.951 | 0.411 | 0.689 | 0.163 | 0.441 | n/a (0) |
| datafog | 0.678 | 0.828 | 0.683 | 0.507 | 0.659 | 0.683 (7) |

## tuned track, shared view, privacy-real

| entry | mR | mP | mF2 | doc leak | pF1 | mF2 out-of-domain (n sets) |
|---|---|---|---|---|---|---|
| datafog | 0.768 | 0.750 | 0.750 | 0.812 | 0.819 | 0.750 (2) |
| sct-default | 0.983 | 0.193 | 0.536 | 0.180 | 0.298 | n/a (0) |
| sct-gliner | 0.976 | 0.177 | 0.509 | 0.188 | 0.292 | n/a (0) |
| opf | 0.547 | 0.462 | 0.425 | 0.569 | 0.508 | n/a (0) |
| bardsai-v3 | 0.455 | 0.193 | 0.351 | 0.646 | 0.450 | 0.351 (2) |
| gliner2-fastino | 0.527 | 0.161 | 0.348 | 0.599 | 0.480 | 0.348 (2) |
| presidio | 0.456 | 0.172 | 0.323 | 0.721 | 0.363 | 0.323 (2) |
| gliner-nvidia | 0.308 | 0.367 | 0.269 | 0.837 | 0.335 | 0.269 (2) |

## tuned track, shared view, ner

| entry | mR | mP | mF2 | doc leak | pF1 | mF2 out-of-domain (n sets) |
|---|---|---|---|---|---|---|
| bardsai-v3 | 0.500 | 0.100 | 0.119 | 0.500 | 0.500 | 0.119 (5) |
| gliner2-fastino | 0.917 | 0.030 | 0.090 | 0.125 | 0.880 | 0.090 (5) |
| sct-default | 1.000 | 0.026 | 0.085 | 0.000 | 0.015 | n/a (0) |
| presidio | 0.583 | 0.040 | 0.084 | 0.625 | 0.522 | 0.084 (5) |
| opf | 0.833 | 0.023 | 0.075 | 0.250 | 0.727 | n/a (0) |
| sct-gliner | 1.000 | 0.021 | 0.073 | 0.000 | 0.030 | n/a (0) |
| gliner-nvidia | 0.833 | 0.019 | 0.065 | 0.250 | 0.138 | 0.065 (5) |
| datafog | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 (5) |

## tuned track, full view, privacy-synthetic

| entry | mR | mP | mF2 | doc leak | pF1 | mF2 out-of-domain (n sets) |
|---|---|---|---|---|---|---|
| bardsai-v3 | 0.908 | 0.917 | 0.910 | 0.214 | 0.858 | 0.814 (3) |
| sct-default | 0.927 | 0.756 | 0.878 | 0.363 | 0.597 | 0.948 (2) |
| gliner2-fastino | 0.909 | 0.756 | 0.868 | 0.335 | 0.773 | 0.868 (7) |
| gliner-nvidia | 0.870 | 0.821 | 0.856 | 0.364 | 0.700 | 0.743 (3) |
| opf | 0.803 | 0.842 | 0.806 | 0.391 | 0.695 | n/a (0) |
| sct-gliner | 0.878 | 0.613 | 0.788 | 0.450 | 0.442 | n/a (0) |
| presidio | 0.705 | 0.730 | 0.697 | 0.645 | 0.557 | 0.697 (7) |
| datafog | 0.337 | 0.854 | 0.379 | 0.919 | 0.377 | 0.379 (7) |

## tuned track, full view, privacy-real

| entry | mR | mP | mF2 | doc leak | pF1 | mF2 out-of-domain (n sets) |
|---|---|---|---|---|---|---|
| sct-default | 0.933 | 0.523 | 0.804 | 0.823 | 0.562 | n/a (0) |
| gliner2-fastino | 0.895 | 0.550 | 0.792 | 0.821 | 0.638 | 0.792 (2) |
| bardsai-v3 | 0.720 | 0.809 | 0.719 | 0.833 | 0.671 | 0.719 (2) |
| sct-gliner | 0.750 | 0.472 | 0.663 | 0.911 | 0.392 | n/a (0) |
| presidio | 0.532 | 0.631 | 0.517 | 0.951 | 0.440 | 0.517 (2) |
| opf | 0.520 | 0.775 | 0.513 | 0.867 | 0.540 | n/a (0) |
| gliner-nvidia | 0.456 | 0.831 | 0.487 | 0.904 | 0.416 | 0.487 (2) |
| datafog | 0.161 | 0.765 | 0.183 | 0.997 | 0.235 | 0.183 (2) |

## tuned track, full view, ner

| entry | mR | mP | mF2 | doc leak | pF1 | mF2 out-of-domain (n sets) |
|---|---|---|---|---|---|---|
| sct-default | 0.855 | 0.692 | 0.812 | 0.351 | 0.784 | n/a (0) |
| gliner2-fastino | 0.776 | 0.714 | 0.747 | 0.344 | 0.624 | 0.747 (5) |
| gliner-nvidia | 0.400 | 0.577 | 0.409 | 0.733 | 0.260 | 0.409 (5) |
| sct-gliner | 0.335 | 0.372 | 0.329 | 0.814 | 0.174 | n/a (0) |
| bardsai-v3 | 0.256 | 0.921 | 0.293 | 0.872 | 0.302 | 0.293 (5) |
| opf | 0.253 | 0.750 | 0.281 | 0.857 | 0.198 | n/a (0) |
| presidio | 0.095 | 0.469 | 0.105 | 0.962 | 0.045 | 0.105 (5) |
| datafog | 0.001 | 0.333 | 0.001 | 1.000 | 0.000 | 0.001 (5) |

## tuned track, per dataset, full view, masked F2 (completed docs)

| dataset | bardsai-v3 | datafog | gliner-nvidia | gliner2-fastino | opf | presidio | sct-default | sct-gliner |
|---|---|---|---|---|---|---|---|---|
| openpii | 0.969 | 0.338 | 0.957 | 0.891 | 0.883 | 0.647 | 0.908 | 0.858 |
| gretel | 0.660 | 0.150 | 0.717 | 0.688 | 0.570 | 0.472 | 0.576 | 0.568 |
| gretel_en | 0.994 | 0.487 | 0.957 | 0.919 | 0.912 | 0.865 | 0.932 | 0.917 |
| nemotron | 0.979 | 0.357 | 0.916 | 0.873 | 0.681 | 0.780 | 0.870 | 0.818 |
| argilla_pii | 0.983 | 0.233 | 0.934 | 0.845 | 0.799 | 0.880 | 0.963 | 0.901 |
| thai_pii | 0.889 | 0.561 | 0.782 | 0.952 | 0.898 | 0.486 | 0.969 | 0.708 |
| bayanat | 0.893 | 0.527 | 0.731 | 0.920 | 0.898 | 0.751 | 0.926 | 0.742 |
| tab | 0.638 | 0.093 | 0.363 | 0.776 | 0.284 | 0.486 | 0.830 | 0.647 |
| redact | 0.801 | 0.274 | 0.611 | 0.808 | 0.743 | 0.549 | 0.778 | 0.678 |
| iahlt | 0.433 | 0.000 | 0.748 | 0.869 | 0.181 | 0.188 | 0.898 | 0.490 |
| thainer | 0.394 | 0.004 | 0.215 | 0.588 | 0.502 | 0.107 | 0.864 | 0.216 |
| naamapadam | 0.391 | 0.000 | 0.120 | 0.884 | 0.369 | 0.102 | 0.715 | 0.090 |
| multiconer | 0.225 | 0.000 | 0.313 | 0.642 | 0.269 | 0.076 | 0.746 | 0.229 |
| stockmark | 0.024 | 0.000 | 0.648 | 0.750 | 0.085 | 0.051 | 0.837 | 0.620 |

## tuned track, completion

| entry | docs | failed | flagged datasets (< 99% complete) |
|---|---|---|---|
| bardsai-v3 | 16661 | 0 | - |
| datafog | 16661 | 0 | - |
| gliner-nvidia | 16661 | 0 | - |
| gliner2-fastino | 16661 | 13 | - |
| opf | 16661 | 0 | - |
| presidio | 16661 | 0 | - |
| sct-default | 16661 | 0 | - |
| sct-gliner | 16661 | 0 | - |

## tuned track, paired bootstrap vs baseline

| entry | baseline better | entry better | no difference (masked_recall, shared view) |
|---|---|---|---|
| bardsai-v3 | bayanat, redact, tab, thai_pii, thainer | gretel, gretel_en, nemotron, openpii | 5 |
| datafog | argilla_pii, bayanat, gretel, gretel_en, nemotron, openpii, redact, tab, thai_pii, thainer | - | 4 |
| gliner-nvidia | bayanat, redact, tab, thai_pii | gretel, gretel_en, openpii | 7 |
| gliner2-fastino | argilla_pii, bayanat, redact, tab, thai_pii | gretel | 8 |
| opf | argilla_pii, tab, thai_pii | openpii | 10 |
| presidio | argilla_pii, bayanat, gretel_en, nemotron, openpii, redact, tab, thai_pii, thainer | - | 5 |
| sct-gliner | redact | openpii | 12 |

## speed (speed-16t)

device cpu, cpus 0-15, threads 16, 100 docs, 3 repeats, median; peak RSS from getrusage (on Linux it never reads below the runner's own ~2 GB: values at that floor are an upper bound)

| entry | cold start s | latency median ms | p95 ms | batch-8 docs/s | chars/s | peak RSS MB | venv MB | failed |
|---|---|---|---|---|---|---|---|---|
| datafog | 0.101 | 0.32 | 2.2 | 1787 | 3.518e+06 | 2014 | 9.3 | 0 |
| presidio | 1.504 | 20.61 | 118.8 | 27.39 | 5.392e+04 | 2014 | 11398.6 | 0 |
| bardsai-v3 | 3.614 | 252.6 | 1106 | 2.523 | 4966 | 2014 | 5416.3 | 0 |
| sct-default | 8.372 | 610 | 2192 | 1.495 | 2943 | 7692 | 473.7 | 0 |
| sct-gliner | 7.378 | 836.4 | 5656 | 0.529 | 1042 | 6414 | 5769.2 | 0 |
| gliner2-fastino | 6.666 | 196 | 7431 | 0.524 | 1032 | 6832 | 5402.3 | 0 |
| opf | 4.371 | 639.2 | 4913 | 0.293 | 577.1 | 2.611e+04 | 5382.7 | 0 |
| gliner-nvidia | 7.071 | 866.7 | 5298 | 0.148 | 291.4 | 2.573e+04 | 5388.7 | 0 |

## speed (speed-16t-rss)

device cpu, cpus 0-15, threads 16, 100 docs, 3 repeats, median; peak RSS from /proc VmHWM

| entry | cold start s | latency median ms | p95 ms | batch-8 docs/s | chars/s | peak RSS MB | venv MB | failed |
|---|---|---|---|---|---|---|---|---|
| datafog | 0.101 | 0.31 | 2.17 | 1793 | 3.529e+06 | 30.3 | 9.3 | 0 |
| presidio | 1.509 | 20.76 | 119.4 | 27.28 | 5.37e+04 | 826.1 | 11398.6 | 0 |
| bardsai-v3 | 3.624 | 252.6 | 1220 | 2.514 | 4948 | 1738 | 5416.3 | 0 |

## speed (speed-1t)

device cpu, cpus 0, threads 1, 30 docs, 1 repeats, median; peak RSS from /proc VmHWM

| entry | cold start s | latency median ms | p95 ms | batch-8 docs/s | chars/s | peak RSS MB | venv MB | failed |
|---|---|---|---|---|---|---|---|---|
| datafog | 0.104 | 0.29 | 1.21 | 2530 | 3.374e+06 | 30.3 | 9.3 | 0 |
| presidio | 1.513 | 18.21 | 91.13 | 37.77 | 5.037e+04 | 777.6 | 11398.6 | 0 |
| bardsai-v3 | 3.48 | 314.9 | 1951 | 1.65 | 2200 | 1272 | 5416.3 | 0 |
| sct-default | 8.77 | 4244 | 2.014e+04 | 0.194 | 258.6 | 5919 | 474.2 | 0 |
| opf | 4.134 | 701.4 | 8131 | 0.164 | 218.5 | 8406 | 5382.7 | 0 |
| gliner2-fastino | 7.162 | 1429 | 2.656e+04 | 0.156 | 208.5 | 4476 | 5402.3 | 0 |
| sct-gliner | 7.553 | 2953 | 1.825e+04 | 0.145 | 193.1 | 5327 | 5769.2 | 0 |
| gliner-nvidia | 8.377 | 5958 | 3.809e+04 | 0.025 | 33.4 | 1.757e+04 | 5388.7 | 0 |
