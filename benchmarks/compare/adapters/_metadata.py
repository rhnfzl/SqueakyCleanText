"""Frozen model checkpoints and label maps shared by isolated adapter scripts."""

PERSON, LOCATION, ORGANISATION = "PERSON", "LOCATION", "ORGANISATION"
EMAIL, PHONE, IP, URL, DATE, NUMBER = "EMAIL", "PHONE", "IP", "URL", "DATE", "NUMBER"


GLINER_PROMPTS = {
    "person": PERSON, "organization": ORGANISATION, "location": LOCATION, "address": LOCATION,
    "email": EMAIL, "phone number": PHONE, "ip address": IP, "url": URL, "date": DATE,
    "date of birth": DATE, "credit card number": NUMBER, "iban": NUMBER, "social security number": NUMBER,
    "passport number": NUMBER, "account number": NUMBER, "tax id": NUMBER, "national id": NUMBER,
}

NVIDIA_CARD = {
    "first_name": PERSON, "last_name": PERSON, "company_name": ORGANISATION,
    **dict.fromkeys(("street_address", "city", "state", "country"), LOCATION),
    "postcode": NUMBER,  # as gold postcode (nemotron, gretel_en) and Fastino postal_code
    "email": EMAIL, "phone_number": PHONE, "fax_number": PHONE, "ipv4": IP, "ipv6": IP, "url": URL,
    "date": DATE, "date_of_birth": DATE, "date_time": DATE,
    **dict.fromkeys(("account_number", "credit_debit_card", "ssn", "bank_routing_number", "swift_bic"), NUMBER),
}

FASTINO_CARD = {  # credentials and usernames are out of scope (absent)
    **dict.fromkeys(("person", "full_name", "first_name", "middle_name", "last_name"), PERSON),
    **dict.fromkeys(("address", "street_address", "city", "state_or_region", "country"), LOCATION),
    "email": EMAIL, "phone_number": PHONE, "ip_address": IP,
    **dict.fromkeys(("date_of_birth", "sensitive_date", "document_date", "expiration_date",
                     "transaction_date", "card_expiry"), DATE),
    **dict.fromkeys(("postal_code", "government_id", "national_id_number", "passport_number",
                     "drivers_license_number", "license_number", "tax_id", "tax_number", "bank_account",
                     "account_number", "routing_number", "iban", "payment_card", "card_number", "card_cvv",
                     "account_id", "sensitive_account_id"), NUMBER),
}

PIIRANHA = {**dict.fromkeys(("GIVENNAME", "SURNAME"), PERSON),
            **dict.fromkeys(("CITY", "STREET", "BUILDINGNUM"), LOCATION),
            "EMAIL": EMAIL, "TELEPHONENUM": PHONE, "DATEOFBIRTH": DATE,
            **dict.fromkeys(("ACCOUNTNUM", "CREDITCARDNUMBER", "DRIVERLICENSENUM", "IDCARDNUM", "SOCIALNUM", "TAXNUM",
                             "ZIPCODE"), NUMBER)}

AI4PRIVACY = {**dict.fromkeys(("GIVENNAME", "SURNAME"), PERSON),
              **dict.fromkeys(("CITY", "STREET", "BUILDINGNUM"), LOCATION),
              "EMAIL": EMAIL, "TELEPHONENUM": PHONE, "DATE": DATE,
              **dict.fromkeys(("CREDITCARDNUMBER", "DRIVERLICENSENUM", "IDCARDNUM", "PASSPORTNUM", "SOCIALNUM", "TAXNUM",
                               "ZIPCODE"), NUMBER)}

OPENMED = {**dict.fromkeys(("FIRSTNAME", "MIDDLENAME", "LASTNAME"), PERSON),
           **dict.fromkeys(("STREET", "BUILDINGNUMBER", "SECONDARYADDRESS", "CITY", "COUNTY", "STATE", "GPSCOORDINATES"),
                           LOCATION),
           "ORGANIZATION": ORGANISATION, "EMAIL": EMAIL, "PHONE": PHONE, "URL": URL, "IPADDRESS": IP,
           "DATE": DATE, "DATEOFBIRTH": DATE,
           **dict.fromkeys(("BANKACCOUNT", "IBAN", "BIC", "CREDITCARD", "CVV", "PIN", "MASKEDNUMBER", "SSN", "ZIPCODE",
                            "VIN", "VRM", "IMEI", "MACADDRESS"), NUMBER)}

GLINER_NVIDIA_MODEL = 'nvidia/gliner-PII'
GLINER_NVIDIA_REVISION = 'bd23e8ef4425fd04e34c5204ab49ffaa706eae79'

GLINER2_FASTINO_MODEL = 'fastino/gliner2-privacy-filter-PII-multi'
GLINER2_FASTINO_REVISION = '1cb4166094dc58fa8d836429f060d6c95f62b495'

GLINER_KNOWLEDGATOR_MODEL = 'knowledgator/gliner-pii-base-v1.0'
GLINER_KNOWLEDGATOR_REVISION = '61726e0ad791dcab3e29339bbec3ad42ded65641'

OPF_MODEL = 'openai/privacy-filter'
OPF_REVISION = '7ffa9a043d54d1be65afb281eddf0ffbe629385b'

BARDSAI_V3_MODEL = 'bardsai/eu-pii-anonimization-multilang-v3'
BARDSAI_V3_REVISION = '091984b5fbce4c161abc4776ed0f4e3206e1b374'

PIIRANHA_MODEL = 'iiiorg/piiranha-v1-detect-personal-information'
PIIRANHA_REVISION = '255acde67a2f34cf452eb42e365b24d2957352fc'

AI4PRIVACY_MODEL = 'ai4privacy/llama-ai4privacy-multilingual-categorical-anonymiser-openpii'
AI4PRIVACY_REVISION = '83ef30d5e7c9d113ad80ce745b564cdd2320c5d5'

OPENMED_MODEL = 'OpenMed/privacy-filter-multilingual-v2'
OPENMED_REVISION = '0d0c0430fa386435ace1d9f842f0f966903a9ac8'

PII_TRACER_MODEL = 'perplexity-ai/PII-Tracer'
PII_TRACER_REVISION = 'd25c16f2e57e321f6d2527715c01df9112f956f5'

BARDSAI = {
    "PERSON_NAME": "PERSON", "PERSON_ALIAS": "PERSON",
    "LOCATION": "LOCATION", "POSTAL_ADDRESS": "LOCATION", "GEO_LOCATION": "LOCATION",
    "ORGANIZATION_NAME": "ORGANISATION",
    "EMAIL_ADDRESS": "EMAIL", "PHONE_NUMBER": "PHONE", "IP_ADDRESS": "IP",
    "URL": "URL", "IDENTIFYING_LINK": "URL",
    "DATE_TIME": "DATE", "DATE_OF_BIRTH": "DATE",
    **{t: "NUMBER" for t in ("PERSON_IDENTIFIER", "ORGANIZATION_IDENTIFIER", "BANK_ACCOUNT_IDENTIFIER",
                             "ACCOUNT_IDENTIFIER", "PAYMENT_CARD", "PAYMENT_CARD_SECURITY",
                             "DOCUMENT_IDENTIFIER", "VEHICLE_IDENTIFIER", "DEVICE_IDENTIFIER")},
}

PII_TRACER = {"private_person": "PERSON", "private_address": "LOCATION", "private_email": "EMAIL",
          "private_phone": "PHONE", "private_url": "URL", "private_date": "DATE", "account_number": "NUMBER"}

LLM_GUARD = {"PERSON": "PERSON", "EMAIL_ADDRESS": "EMAIL", "EMAIL_ADDRESS_RE": "EMAIL", "PHONE_NUMBER": "PHONE",
          "IP_ADDRESS": "IP",
          **dict.fromkeys(("CREDIT_CARD", "CREDIT_CARD_RE", "IBAN_CODE", "US_SSN", "US_SSN_RE", "US_BANK_NUMBER"),
                          "NUMBER")}

SCRUBADUB = {"name": "PERSON", "organization": "ORGANISATION", "location": "LOCATION", "address": "LOCATION",
          "email": "EMAIL", "phone": "PHONE", "url": "URL", "date_of_birth": "DATE",
          **dict.fromkeys(("credit_card", "drivers_licence", "national_insurance_number", "postalcode",
                           "social_security_number", "tax_reference_number", "vehicle_licence_plate"), "NUMBER")}

OPF = {"private_person": PERSON, "private_address": LOCATION, "private_email": EMAIL,
       "private_phone": PHONE, "private_url": URL, "private_date": DATE, "account_number": NUMBER}

FASTINO_CARD_LABELS = {
    **FASTINO_CARD,
    **dict.fromkeys(("username", "password", "secret", "api_key", "access_token", "recovery_code"), None),
}
