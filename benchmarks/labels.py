"""Dataset label -> SCT type, and gold span clean-up.

This is a schema mapping (dataset label names to SCT output types), not detection.
Gold labels missing from a dataset's map are "out of scope": not scored as misses,
and SCT spans that only overlap out-of-scope gold are not scored as false positives.
Mapping follows sct-integration-research REPORT.md section (c), with SCT 0.8 names.
"""

PERSON, LOCATION, ORGANISATION = "PERSON", "LOCATION", "ORGANISATION"
EMAIL, PHONE, IP, URL, DATE, NUMBER = "EMAIL", "PHONE", "IP", "URL", "DATE", "NUMBER"
# Gold with no entity type (RedactionBench): in every view, scored by the type-agnostic measures only.
UNTYPED = "UNTYPED"

# SCT placeholder type -> scored type. YEAR is a date; the rest are pre-0.8 names.
PRED_ALIASES = {"YEAR": DATE, "PHONE_NUMBER": PHONE, "IP_ADDRESS": IP, "IBAN": NUMBER, "CARD": NUMBER, "ID": NUMBER}

# Gold labels that mark one part of a name (given name, surname). Only these merge
# into a full name; two plain PERSON entities next to each other stay two entities.
NAME_PARTS = frozenset({"GIVENNAME", "SURNAME", "first_name", "last_name", "First_Given_Name", "Last_Family_Name",
                        "FIRSTNAME", "MIDDLENAME", "LASTNAME", "FIRST_NAME", "MIDDLE_NAME", "LAST_NAME",
                        "NAME_GIVEN", "NAME_FAMILY"})

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
    "gretel_en": {
        "name": PERSON, "first_name": PERSON, "last_name": PERSON, "company_name": ORGANISATION,
        "street_address": LOCATION, "address": LOCATION, "city": LOCATION, "state": LOCATION,
        "country": LOCATION, "email": EMAIL, "phone_number": PHONE, "ipv4": IP, "ipv6": IP, "url": URL,
        "date": DATE, "date_of_birth": DATE, "date_time": DATE, "ssn": NUMBER, "credit_card_number": NUMBER,
        "account_number": NUMBER, "tax_id": NUMBER, "national_id": NUMBER, "postcode": NUMBER,
    },
    "argilla_pii": {
        "FIRSTNAME": PERSON, "MIDDLENAME": PERSON, "LASTNAME": PERSON, "CITY": LOCATION, "STREET": LOCATION,
        "STATE": LOCATION, "COUNTY": LOCATION, "COMPANYNAME": ORGANISATION, "EMAIL": EMAIL,
        "PHONENUMBER": PHONE, "IP": IP, "IPV4": IP, "IPV6": IP, "URL": URL, "DATE": DATE, "DOB": DATE,
        "ACCOUNTNUMBER": NUMBER, "CREDITCARDNUMBER": NUMBER, "SSN": NUMBER, "IBAN": NUMBER, "ZIPCODE": NUMBER,
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
    # Round 2 (PROTOCOL.md section 1). Postcodes are NUMBER, as in round 1 (openpii ZIPCODE, nemotron postcode).
    "newmean": {
        "PERSON": PERSON, "ADDRESS": LOCATION, "ORG": ORGANISATION, "EMAIL": EMAIL, "PHONE": PHONE, "URL": URL,
        "IP_ADDRESS": IP, "DATE_OF_BIRTH": DATE, "ID_NUMBER": NUMBER, "FINANCIAL": NUMBER,
    },
    "dialogpii": {
        "PERSON": PERSON, "ORG": ORGANISATION, "PERSON_EMAIL": EMAIL, "CODE_PHONE": PHONE, "CODE_URL": URL,
        "DATETIME": DATE, "CODE": NUMBER, "LOC_ZIP": NUMBER,
        **dict.fromkeys(("LOC_CITY", "LOC_COUNTRY", "LOC_STREET", "LOC_HOUSENUMBER", "LOC_OTHER"), LOCATION),
    },
    "kdpii": {
        "PS_NAME": PERSON, "PS_NICKNAME": PERSON,
        **dict.fromkeys(("OG_WORKPLACE", "OGG_EDUCATION", "OG_DEPARTMENT", "OGG_CLUB"), ORGANISATION),
        "LC_ADDRESS": LOCATION, "LC_PLACE": LOCATION, "LCP_COUNTRY": LOCATION, "TMI_EMAIL": EMAIL, "TMI_SITE": URL,
        "QT_IP": IP, "QT_MOBILE": PHONE, "QT_PHONE": PHONE, "DT_BIRTH": DATE,
        **dict.fromkeys(("QT_RESIDENT_NUMBER", "QT_ALIEN_NUMBER", "QT_PASSPORT_NUMBER", "QT_DRIVER_NUMBER",
                         "QT_ACCOUNT_NUMBER", "QT_PLATE_NUMBER", "QT_CARD_NUMBER"), NUMBER),
        # PS_ID (user names) out of scope, like USERNAME in newmean and tonic
    },
    "nus_lt": {
        "PER": PERSON, "LOC": LOCATION, "ORG": ORGANISATION, "DATE": DATE, "NUM_PHONE": PHONE,
        "ID_PER": NUMBER, "ID_MISC": NUMBER, "NUM_CAR": NUMBER, "MISC:email": EMAIL, "MISC:url": URL,
    },
    "redmadrobot": {
        "FIRST_NAME": PERSON, "MIDDLE_NAME": PERSON, "LAST_NAME": PERSON,
        **dict.fromkeys(("COUNTRY", "REGION", "DISTRICT", "CITY", "STREET", "HOUSE"), LOCATION),
        "EMAIL": EMAIL, "PHONE": PHONE, "URL": URL, "IP_ADDRESS": IP,
        **dict.fromkeys(("CREDIT_CARD", "PASSPORT", "INN", "SNILS", "OMS", "DRIVER_LICENSE", "MILITARY_ID",
                         "BIRTH_CERTIFICATE"), NUMBER),
    },
    "redactionbench": {"mandatory": UNTYPED},
    "tonic": {
        "NAME_GIVEN": PERSON, "NAME_FAMILY": PERSON, "ORGANIZATION": ORGANISATION, "EMAIL_ADDRESS": EMAIL,
        "PHONE_NUMBER": PHONE, "URL": URL, "LOCATION_ADDRESS": LOCATION, "EMPLOYEE_ID": NUMBER, "ACCOUNT_NUMBER": NUMBER,
    },
    "meddeid": {
        "Name": PERSON, "Organization": ORGANISATION, "Address_Location": LOCATION, "Date": DATE,
        "Age_Birthdate:date": DATE, "Contactdetails:email": EMAIL, "Contactdetails:phone": PHONE, "ID": NUMBER,
    },
    "greek_deid": {
        "greek_patient_name": PERSON, "greek_clinician": PERSON, "hospital": ORGANISATION, "url": URL,
        "email": EMAIL, "address": LOCATION, "greek_phone": PHONE, "date": DATE, "dob": DATE,
        "amka": NUMBER, "gr_id": NUMBER,
    },
    "neonredact": {
        "KISI": PERSON, "ADRES": LOCATION, "KURUM": ORGANISATION, "TELEFON": PHONE, "EPOSTA": EMAIL,
        "DOGUM_TARIHI": DATE, "OLAY_TARIHI": DATE,
        **dict.fromkeys(("TC_KIMLIK", "VKN", "IBAN", "KART", "PLAKA"), NUMBER),
    },
    "pl_pii": {
        "PERSON": PERSON, "ORG": ORGANISATION, "LOC": LOCATION, "POSTAL": NUMBER, "DOB": DATE, "PHONE": PHONE,
        "EMAIL": EMAIL,
        **dict.fromkeys(("PESEL", "NIP", "REGON", "DOWOD", "IBAN", "PASSPORT", "DRIVING_LICENSE", "PAYMENT_CARD",
                         "VIN", "VEHICLE_PLATE"), NUMBER),
    },
    "europii": {
        "PERSON": PERSON, "EMAIL": EMAIL, "PHONE": PHONE, "IP_ADDRESS": IP,
        **dict.fromkeys(("IBAN", "PAYMENT_CARD", "VAT_ID", "NATIONAL_ID"), NUMBER),
    },
}

# Round-2 gold labels with no SCT type, named from each card / verify.md (labels only, never values). A round-2
# gold label in neither the map nor this list stops the run (``unknown_gold_labels``): it would otherwise drop out
# of scope silently and move precision and the recall base. A new one is added here or to the map, by hand.
DATASET_OUT_OF_SCOPE: dict[str, frozenset] = {
    "newmean": frozenset({"SECRET", "USERNAME"}),
    "dialogpii": frozenset({"PERSON_SOCIAL_RELATION", "PROFESSION", "QUANTITY", "PRODUCT", "DATETIME_AGE", "MISC"}),
    "kdpii": frozenset({"PS_ID", "OGG_RELIGION", "CV_POSITION", "CV_SEX", "CV_MILITARY_CAMP", "QT_AGE", "QT_LENGTH",
                        "QT_WEIGHT", "QT_GRADE", "FD_MAJOR", "TM_BLOOD_TYPE"}),
    # the card's 8 sensitive categories are not named in verify.md: the first load names any that occur
    "nus_lt": frozenset({"MISC", "TIME", "DURATION", "VALUE", "AGE", "OCC", "EDU", "NAT"}),
    "redmadrobot": frozenset(),
    "redactionbench": frozenset({"contextual"}),
    "tonic": frozenset({"USERNAME"}),
    "meddeid": frozenset({"Profession", "Age_Birthdate:age"}),
    "greek_deid": frozenset(),
    "neonredact": frozenset({"MESLEK", "AKRABALIK", "SAGLIK", "TARAF_SIFATI", "GIZLI", "SICIL_NO", "SASI_NO",
                             "MOTOR_NO", "EHLIYET_NO", "BELGE_NO", "KULLANICI_ADI", "RUHSAT_NO"}),
    "pl_pii": frozenset(f"{label}:keep" for label in DATASET_LABELS["pl_pii"]),
    "europii": frozenset(),
}


def unknown_gold_labels(dataset: str, records: list[dict]) -> dict[str, int]:
    """Gold label -> count for labels neither mapped nor declared out of scope (datasets with a declared list)."""
    if dataset not in DATASET_OUT_OF_SCOPE:
        return {}
    known = DATASET_LABELS[dataset].keys() | DATASET_OUT_OF_SCOPE[dataset]
    counts: dict[str, int] = {}
    for r in records:
        for s in r["spans"]:
            if s["label"] not in known:
                counts[s["label"]] = counts.get(s["label"], 0) + 1
    return counts


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
