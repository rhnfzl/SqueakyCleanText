"""Dataset label -> SCT type, and gold span clean-up.

This is a schema mapping (dataset label names to SCT output types), not detection.
Gold labels missing from a dataset's map are "out of scope": not scored as misses,
and SCT spans that only overlap out-of-scope gold are not scored as false positives.
Mapping follows sct-integration-research REPORT.md section (c), with SCT 0.8 names.
"""

PERSON, LOCATION, ORGANISATION = "PERSON", "LOCATION", "ORGANISATION"
EMAIL, PHONE, IP, URL, DATE, NUMBER = "EMAIL", "PHONE", "IP", "URL", "DATE", "NUMBER"

# SCT placeholder type -> scored type. YEAR is a date; the rest are pre-0.8 names.
PRED_ALIASES = {"YEAR": DATE, "PHONE_NUMBER": PHONE, "IP_ADDRESS": IP}

# Gold labels that mark one part of a name (given name, surname). Only these merge
# into a full name; two plain PERSON entities next to each other stay two entities.
NAME_PARTS = frozenset({"GIVENNAME", "SURNAME", "first_name", "last_name", "First_Given_Name", "Last_Family_Name"})

_PER_LOC_ORG = {"PER": PERSON, "LOC": LOCATION, "ORG": ORGANISATION}

DATASET_LABELS: dict[str, dict[str, str]] = {
    "openpii": {
        "GIVENNAME": PERSON, "SURNAME": PERSON, "CITY": LOCATION, "STREET": LOCATION,
        "EMAIL": EMAIL, "TELEPHONENUM": PHONE, "DATE": DATE,
        "IDCARDNUM": NUMBER, "CREDITCARDNUMBER": NUMBER, "SOCIALNUM": NUMBER, "TAXNUM": NUMBER,
        "PASSPORTNUM": NUMBER, "DRIVERLICENSENUM": NUMBER, "ZIPCODE": NUMBER,
    },
    "gretel": {
        "name": PERSON, "first_name": PERSON, "last_name": PERSON, "company": ORGANISATION,
        "street_address": LOCATION, "email": EMAIL, "phone_number": PHONE, "ipv4": IP, "ipv6": IP,
        "date": DATE, "date_of_birth": DATE, "iban": NUMBER, "bban": NUMBER,
        "credit_card_number": NUMBER, "ssn": NUMBER, "passport_number": NUMBER,
        "driver_license_number": NUMBER,
    },
    "nemotron": {
        "first_name": PERSON, "last_name": PERSON, "city": LOCATION, "county": LOCATION,
        "state": LOCATION, "country": LOCATION, "street_address": LOCATION, "company_name": ORGANISATION,
        "email": EMAIL, "phone_number": PHONE, "fax_number": PHONE, "ipv4": IP, "ipv6": IP, "url": URL,
        "date": DATE, "date_of_birth": DATE, "ssn": NUMBER, "credit_debit_card": NUMBER,
        "account_number": NUMBER, "tax_id": NUMBER, "postcode": NUMBER,
    },
    "redact": {
        "Full_Name": PERSON, "First_Given_Name": PERSON, "Last_Family_Name": PERSON, "Preferred_Name": PERSON,
        "City": LOCATION, "State": LOCATION, "Location": LOCATION, "Address_Personal": LOCATION,
        "Address_Work": LOCATION, "Place_of_Birth": LOCATION, "Country_of_Residence": LOCATION,
        "Org_Name": ORGANISATION, "Personal_Email_Address": EMAIL, "Work_Email_Address": EMAIL,
        "Telephone_Numbers_Personal": PHONE, "Telephone_Numbers_Work": PHONE, "Static_IP_Address": IP,
        "Date_Time": DATE, "Date_of_Birth": DATE, "Credit_Card_Numbers": NUMBER,
        "National_Identification_Number": NUMBER, "Passport_Number": NUMBER,
        "Tax_Reference_Number": NUMBER, "Driving_License_Number": NUMBER,
    },
    "iahlt": {"PER": PERSON, "GPE": LOCATION, "LOC": LOCATION, "FAC": LOCATION, "ORG": ORGANISATION},
    "bayanat": {
        "PERSON": PERSON, "PHONE_NUMBER": PHONE, "EMAIL_ADDRESS": EMAIL, "URL": URL, "IBAN_CODE": NUMBER,
        "SA_NATIONAL_ID": NUMBER, "SA_IQAMA": NUMBER, "EG_NATIONAL_ID": NUMBER, "KW_CIVIL_ID": NUMBER,
        "AE_EMIRATES_ID": NUMBER, "BH_CPR": NUMBER,
    },
    "thainer": {
        "PERSON": PERSON, "LOCATION": LOCATION, "FACILITY": LOCATION, "ORGANIZATION": ORGANISATION,
        "EMAIL": EMAIL, "PHONE": PHONE, "URL": URL, "DATE": DATE, "ZIP": NUMBER,
    },
    "thai_pii": {
        "NAME": PERSON, "ADDRESS": LOCATION, "BANK": ORGANISATION, "EMAIL": EMAIL, "PHONE": PHONE,
        "ID_CARD": NUMBER, "BANK_ACC": NUMBER,
    },
    "tab": {"PERSON": PERSON, "LOC": LOCATION, "ORG": ORGANISATION, "DATETIME": DATE, "CODE": NUMBER},
    "naamapadam": dict(_PER_LOC_ORG),
    "multiconer": {
        **dict.fromkeys(("Artist", "Athlete", "Politician", "Cleric", "Scientist", "SportsManager",
                         "OtherPER"), PERSON),
        **dict.fromkeys(("Facility", "OtherLOC", "HumanSettlement", "Station"), LOCATION),
        **dict.fromkeys(("MusicalGRP", "PublicCorp", "PrivateCorp", "AerospaceManufacturer", "SportsGRP",
                         "CarManufacturer", "ORG"), ORGANISATION),
    },
    "stockmark": {
        "人名": PERSON, "地名": LOCATION, "施設名": LOCATION,
        "法人名": ORGANISATION, "政治的組織名": ORGANISATION, "その他の組織名": ORGANISATION,
    },
}


def normalise_gold(text: str, spans: list[dict], mapping: dict[str, str]) -> tuple[list[dict], list[dict]]:
    """Return ``(in_scope, out_of_scope)`` gold spans; in-scope ones carry SCT labels.

    Adjacent name-part spans (``NAME_PARTS``: GIVENNAME + SURNAME) with only whitespace
    between them merge into one name, since SCT tags a full name as one span. A span inside
    another span of the same type is dropped, and duplicates are removed.
    """
    in_scope, out_of_scope = [], []
    for span in spans:
        label = mapping.get(span["label"])
        if label:
            in_scope.append({"start": span["start"], "end": span["end"], "label": label,
                             "part": span["label"] in NAME_PARTS})
        else:
            out_of_scope.append({"start": span["start"], "end": span["end"], "label": span["label"]})
    in_scope.sort(key=lambda s: (s["start"], -s["end"]))
    merged: list[dict] = []
    for span in in_scope:
        previous = next((m for m in reversed(merged) if m["label"] == span["label"]), None)
        if previous and span["end"] <= previous["end"]:
            continue  # nested in (or equal to) a same-type span
        if (previous and span["label"] == PERSON and previous["part"] and span["part"]
                and previous["end"] <= span["start"] and not text[previous["end"]:span["start"]].strip()):
            previous["end"] = span["end"]
            continue
        merged.append(dict(span))
    return [{k: v for k, v in m.items() if k != "part"} for m in merged], out_of_scope
