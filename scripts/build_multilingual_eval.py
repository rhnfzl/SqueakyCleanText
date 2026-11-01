#!/usr/bin/env python3
"""Build the synthetic multilingual PII eval set ``evaluation/multilingual_pii.jsonl``.

Native-script templates with ``{LABEL}`` slots are filled from per-language value
lists. Entity spans come from where each value is inserted, never hand counted.
The seed is fixed, so the output is byte-identical on every run:

    python -m scripts.build_multilingual_eval

Labels match ``evaluation/release_benchmark.jsonl`` and SCT's placeholders:
PERSON, LOCATION, ORGANISATION, EMAIL, PHONE, IP. In non-Latin languages
some PERSON slots get a Latin-script name (slice ``code-switch``).
"""

import argparse
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "evaluation" / "multilingual_pii.jsonl"
SEED = 8
RECORDS_PER_LANGUAGE = 20
CODE_SWITCH_SHARE = 0.25
LABELS = ("PERSON", "LOCATION", "ORGANISATION", "EMAIL", "PHONE", "IP")
SLOT = re.compile(r"\{([A-Z_]+)\}")
DOMAINS = ("support", "hr", "security", "office", "billing")
# Documentation ranges only (RFC 5737, RFC 3849).
IPS = ("192.0.2.17", "198.51.100.42", "203.0.113.5", "192.0.2.201", "2001:db8:85a3::8a2e:370:7334")

# One template per entry of DOMAINS, in that order.
LANGUAGES: dict[str, dict] = {
    "en": {
        "templates": (
            "Please contact {PERSON} at {EMAIL} or call {PHONE}.",
            "{PERSON} moved to {LOCATION} last spring to work for {ORGANISATION}.",
            "The login from {IP} was blocked by {ORGANISATION} after a report from {PERSON}.",
            "Our office in {LOCATION} is run by {PERSON}; the front desk is on {PHONE}.",
            "Invoice questions for {ORGANISATION} go to {EMAIL}, attention {PERSON}.",
        ),
        "PERSON": ("Maria Chen", "John Smith", "Aisha Bello", "Daniel Okafor", "Emily Walker"),
        "LOCATION": ("Manchester", "Toronto", "Cape Town", "Brisbane", "Dublin"),
        "ORGANISATION": ("Northwind Traders", "Bluefield Bank", "Harbor Logistics", "Contoso Health"),
        "EMAIL": ("m.chen@example.com", "j.smith@example.co.uk", "aisha.bello@example.org", "billing@example.net"),
        "PHONE": ("+44 20 7946 0958", "+1-202-555-0147", "(416) 555-0199", "+61 2 5550 1234"),
    },
    "nl": {
        "templates": (
            "Neem contact op met {PERSON} via {EMAIL} of bel {PHONE}.",
            "{PERSON} verhuisde vorig jaar naar {LOCATION} om bij {ORGANISATION} te werken.",
            "De inlog vanaf {IP} is door {ORGANISATION} geblokkeerd na een melding van {PERSON}.",
            "Ons kantoor in {LOCATION} wordt geleid door {PERSON}; de balie is bereikbaar op {PHONE}.",
            "Vragen over facturen van {ORGANISATION} kunt u mailen naar {EMAIL}, t.a.v. {PERSON}.",
        ),
        "PERSON": ("Jan de Vries", "Sanne Jansen", "Pieter Bakker", "Fatima El Amrani", "Lotte Visser"),
        "LOCATION": ("Utrecht", "Groningen", "Eindhoven", "Maastricht", "Zwolle"),
        "ORGANISATION": ("Bakker Logistiek B.V.", "Stichting Zorgnet", "Van Dijk Advocaten", "Polderbank"),
        "EMAIL": ("jan.devries@example.nl", "s.jansen@example.nl", "facturen@example.com", "lotte@example.org"),
        "PHONE": ("+31 6 12345678", "06-12345678", "+31 20 123 4567", "030-1234567"),
    },
    "de": {
        "templates": (
            "Bitte wenden Sie sich an {PERSON} unter {EMAIL} oder rufen Sie {PHONE} an.",
            "{PERSON} ist letztes Jahr nach {LOCATION} gezogen, um bei {ORGANISATION} zu arbeiten.",
            "Die Anmeldung von {IP} wurde von {ORGANISATION} nach einem Hinweis von {PERSON} gesperrt.",
            "Unser Büro in {LOCATION} leitet {PERSON}; die Zentrale erreichen Sie unter {PHONE}.",
            "Rechnungsfragen an {ORGANISATION} bitte an {EMAIL}, z. Hd. {PERSON}.",
        ),
        "PERSON": ("Lukas Müller", "Anna Schneider", "Jonas Weber", "Mehmet Yılmaz", "Sophie Fischer"),
        "LOCATION": ("München", "Hamburg", "Köln", "Leipzig", "Freiburg"),
        "ORGANISATION": ("Müller & Söhne GmbH", "Stadtwerke Bergheim", "Nordlicht Versicherung AG", "Kanzlei Weber"),
        "EMAIL": ("l.mueller@example.de", "anna.schneider@example.de", "rechnung@example.com", "info@example.org"),
        "PHONE": ("+49 30 12345678", "0171 2345678", "+49 89 1234 5678", "040 123456"),
    },
    "es": {
        "templates": (
            "Póngase en contacto con {PERSON} en {EMAIL} o llame al {PHONE}.",
            "{PERSON} se mudó a {LOCATION} el año pasado para trabajar en {ORGANISATION}.",
            "El acceso desde {IP} fue bloqueado por {ORGANISATION} tras un aviso de {PERSON}.",
            "Nuestra oficina de {LOCATION} la dirige {PERSON}; la recepción atiende en el {PHONE}.",
            "Las dudas sobre facturas de {ORGANISATION} se envían a {EMAIL}, a la atención de {PERSON}.",
        ),
        "PERSON": ("María García", "José Martínez", "Lucía Fernández", "Carlos Ruiz", "Valentina Gómez"),
        "LOCATION": ("Sevilla", "Valencia", "Bilbao", "Zaragoza", "Málaga"),
        "ORGANISATION": ("Grupo Alameda", "Banco del Sur", "Transportes Ruiz S.L.", "Fundación Mar Azul"),
        "EMAIL": ("maria.garcia@example.es", "jmartinez@example.es", "facturas@example.com", "lucia@example.org"),
        "PHONE": ("+34 612 345 678", "912 345 678", "+34 91 123 45 67", "699 123 456"),
    },
    "zh": {
        "templates": (
            "请联系{PERSON}，邮箱：{EMAIL}，电话：{PHONE}。",
            "{PERSON}去年搬到{LOCATION}，现在在{ORGANISATION}工作。",
            "来自{IP}的登录已被{ORGANISATION}拦截，举报人是{PERSON}。",
            "我们在{LOCATION}的办公室由{PERSON}负责，前台电话是{PHONE}。",
            "关于{ORGANISATION}发票的问题请发送至{EMAIL}，收件人：{PERSON}。",
        ),
        "PERSON": ("张伟", "王芳", "李娜", "刘洋", "陈静"),
        "LATIN_PERSON": ("Michael Brown", "Anna Schmidt"),
        "LOCATION": ("上海", "成都", "杭州", "深圳", "西安"),
        "ORGANISATION": ("华信科技有限公司", "东方物流集团", "长江银行", "明德医院"),
        "EMAIL": ("zhang.wei@example.cn", "wangfang@example.com", "invoice@example.cn", "lina@example.org"),
        "PHONE": ("+86 138 0013 8000", "010-6552 1234", "139-1234-5678", "+86 21 5555 0199"),
    },
    "ja": {
        "templates": (
            "{PERSON}さんへのお問い合わせは{EMAIL}または{PHONE}までお願いします。",
            "{PERSON}は昨年{LOCATION}に引っ越し、{ORGANISATION}で働いています。",
            "{IP}からのログインは{PERSON}の報告を受けて{ORGANISATION}がブロックしました。",
            "{LOCATION}事務所の責任者は{PERSON}で、受付の電話番号は{PHONE}です。",
            "{ORGANISATION}の請求書に関するお問い合わせは{EMAIL}（担当：{PERSON}）までご連絡ください。",
        ),
        "PERSON": ("佐藤健", "鈴木花子", "高橋大輔", "田中美咲", "渡辺翔"),
        "LATIN_PERSON": ("David Miller", "Laura Rossi"),
        "LOCATION": ("大阪", "札幌", "福岡", "京都", "名古屋"),
        "ORGANISATION": ("青葉商事株式会社", "東和銀行", "桜井物流", "みなと総合病院"),
        "EMAIL": ("k.sato@example.jp", "hanako@example.co.jp", "seikyu@example.com", "info@example.org"),
        "PHONE": ("090-1234-5678", "03-1234-5678", "+81 90 1234 5678", "06-6123-4567"),
    },
    "th": {
        "templates": (
            "กรุณาติดต่อคุณ{PERSON} ทางอีเมล {EMAIL} หรือโทร {PHONE}",
            "{PERSON} ย้ายไป{LOCATION}เมื่อปีที่แล้วเพื่อทำงานที่{ORGANISATION}",
            "การเข้าสู่ระบบจาก {IP} ถูก{ORGANISATION}บล็อกหลังจากได้รับแจ้งจาก{PERSON}",
            "สำนักงานที่{LOCATION}มี{PERSON}เป็นผู้ดูแล ติดต่อแผนกต้อนรับได้ที่ {PHONE}",
            "สอบถามเรื่องใบแจ้งหนี้ของ{ORGANISATION} ส่งอีเมลไปที่ {EMAIL} ถึงคุณ{PERSON}",
        ),
        "PERSON": ("สมชาย ใจดี", "สุดารัตน์ วงศ์ทอง", "ประเสริฐ ศรีสุข", "นภา แก้วมณี", "วิชัย บุญมา"),
        "LATIN_PERSON": ("Peter Jones", "Sarah Lee"),
        "LOCATION": ("เชียงใหม่", "ขอนแก่น", "ภูเก็ต", "หาดใหญ่", "นครราชสีมา"),
        "ORGANISATION": ("บริษัท สยามโลจิสติกส์ จำกัด", "ธนาคารกรุงทอง", "โรงพยาบาลเมตตา", "มูลนิธิรักษ์น้ำ"),
        "EMAIL": ("somchai@example.co.th", "sudarat.w@example.com", "billing@example.co.th", "napa@example.org"),
        "PHONE": ("081 234 5678", "02-123-4567", "+66 81 234 5678", "089-765-4321"),
    },
    "ar": {
        "templates": (
            "يرجى التواصل مع {PERSON} عبر البريد {EMAIL} أو الاتصال على {PHONE}.",
            "انتقل {PERSON} إلى {LOCATION} العام الماضي للعمل في {ORGANISATION}.",
            "حظرت {ORGANISATION} تسجيل الدخول من {IP} بعد بلاغ من {PERSON}.",
            "يدير {PERSON} مكتبنا في {LOCATION}، ويمكن الوصول إلى الاستقبال على {PHONE}.",
            "ترسل استفسارات فواتير {ORGANISATION} إلى {EMAIL}، لعناية {PERSON}.",
        ),
        "PERSON": ("محمد العلي", "فاطمة الزهراء", "أحمد حسن", "ليلى منصور", "يوسف خالد"),
        "LATIN_PERSON": ("James Wilson", "Elena Petrova"),
        "LOCATION": ("دبي", "عمّان", "الرياض", "القاهرة", "الدار البيضاء"),
        "ORGANISATION": ("شركة النخيل للتجارة", "بنك الواحة", "مستشفى الأمل", "مجموعة الفجر للنقل"),
        "EMAIL": ("m.alali@example.ae", "fatima@example.com", "billing@example.sa", "info@example.org"),
        "PHONE": ("+971 50 123 4567", "+962 7 9123 4567", "050 123 4567", "+20 10 1234 5678"),
    },
    "hi": {
        "templates": (
            "कृपया {PERSON} से {EMAIL} पर संपर्क करें या {PHONE} पर कॉल करें।",
            "{PERSON} पिछले साल {LOCATION} चले गए और अब {ORGANISATION} में काम करते हैं।",
            "{IP} से हुए लॉगिन को {PERSON} की शिकायत के बाद {ORGANISATION} ने रोक दिया।",
            "{LOCATION} कार्यालय के प्रमुख {PERSON} हैं; रिसेप्शन का नंबर {PHONE} है।",
            "{ORGANISATION} के बिल से जुड़े सवाल {EMAIL} पर भेजें, ध्यानार्थ: {PERSON}।",
        ),
        "PERSON": ("राहुल शर्मा", "प्रिया वर्मा", "अमित कुमार", "सुनीता यादव", "अंजलि गुप्ता"),
        "LATIN_PERSON": ("Rohan Mehta", "Emma Clarke"),
        "LOCATION": ("लखनऊ", "जयपुर", "पुणे", "भोपाल", "वाराणसी"),
        "ORGANISATION": ("सूर्या लॉजिस्टिक्स प्राइवेट लिमिटेड", "गंगा सहकारी बैंक", "आरोग्य अस्पताल", "नवभारत टेक्नोलॉजीज"),
        "EMAIL": ("rahul.sharma@example.in", "priya@example.com", "billing@example.co.in", "amit.k@example.org"),
        "PHONE": ("+91 98765 43210", "98765 43210", "011-2345 6789", "+91 22 2345 6789"),
    },
    "bn": {
        "templates": (
            "অনুগ্রহ করে {PERSON}-এর সাথে {EMAIL} ঠিকানায় যোগাযোগ করুন বা {PHONE} নম্বরে ফোন করুন।",
            "{PERSON} গত বছর {LOCATION} চলে গেছেন এবং এখন {ORGANISATION}-এ কাজ করেন।",
            "{PERSON}-এর অভিযোগের পর {ORGANISATION} {IP} থেকে লগইন বন্ধ করেছে।",
            "{LOCATION} অফিসের দায়িত্বে আছেন {PERSON}; অভ্যর্থনার নম্বর {PHONE}।",
            "{ORGANISATION}-এর বিল সংক্রান্ত প্রশ্ন {EMAIL} ঠিকানায় পাঠান, প্রাপক: {PERSON}।",
        ),
        "PERSON": ("রহিম উদ্দিন", "সুমাইয়া আক্তার", "অনিক রায়", "তানিয়া ইসলাম", "সৌরভ দাস"),
        "LATIN_PERSON": ("Arif Hossain", "Grace Turner"),
        "LOCATION": ("ঢাকা", "চট্টগ্রাম", "কলকাতা", "সিলেট", "খুলনা"),
        "ORGANISATION": ("পদ্মা লজিস্টিকস লিমিটেড", "সোনালী নদী ব্যাংক", "আশা হাসপাতাল", "বাংলা টেক সলিউশনস"),
        "EMAIL": ("rahim.uddin@example.com.bd", "sumaiya@example.com", "bill@example.com.bd", "anik@example.org"),
        "PHONE": ("+880 1712-345678", "01712-345678", "+880 2 9123456", "01819 123456"),
    },
    "ta": {
        "templates": (
            "தயவுசெய்து {PERSON} அவர்களை {EMAIL} மூலம் தொடர்பு கொள்ளவும் அல்லது {PHONE} என்ற எண்ணில் அழைக்கவும்.",
            "{PERSON} கடந்த ஆண்டு {LOCATION} நகருக்குச் சென்று இப்போது {ORGANISATION} இல் பணிபுரிகிறார்.",
            "{PERSON} அளித்த புகாரைத் தொடர்ந்து {IP} இலிருந்து வந்த உள்நுழைவை {ORGANISATION} தடுத்தது.",
            "{LOCATION} அலுவலகத்தின் பொறுப்பாளர் {PERSON}; வரவேற்பு எண் {PHONE}.",
            "{ORGANISATION} கட்டணம் தொடர்பான கேள்விகளை {EMAIL} முகவரிக்கு அனுப்பவும், கவனத்திற்கு: {PERSON}.",
        ),
        "PERSON": ("முருகன் செல்வம்", "லக்ஷ்மி ராஜன்", "கார்த்திக் சுப்பிரமணியன்", "பிரியா கண்ணன்", "அருண் குமார்"),
        "LATIN_PERSON": ("Vijay Anand", "Sophie Martin"),
        "LOCATION": ("சென்னை", "மதுரை", "கோயம்புத்தூர்", "திருச்சி", "சேலம்"),
        "ORGANISATION": ("காவேரி லாஜிஸ்டிக்ஸ் நிறுவனம்", "தமிழ் கூட்டுறவு வங்கி", "நலம் மருத்துவமனை", "வைகை டெக்னாலஜிஸ்"),
        "EMAIL": ("murugan.s@example.in", "lakshmi@example.com", "billing@example.co.in", "arun.k@example.org"),
        "PHONE": ("+91 94440 12345", "044-2345 6789", "94440 12345", "+91 422 234 5678"),
    },
}


def fill(template: str, values: dict[str, str]) -> tuple[str, list[dict]]:
    """Insert slot values into *template*; return the text and the spans of the inserted values."""
    parts, entities, length, last = [], [], 0, 0
    for match in SLOT.finditer(template):
        literal = template[last:match.start()]
        value = values[match.group(1)]
        parts += [literal, value]
        length += len(literal)
        entities.append({"label": match.group(1), "start": length, "end": length + len(value)})
        length += len(value)
        last = match.end()
    parts.append(template[last:])
    return "".join(parts), entities


def build(seed: int = SEED, per_language: int = RECORDS_PER_LANGUAGE) -> list[dict]:
    rng = random.Random(seed)  # noqa: S311  # reproducible test data, not crypto
    records = []
    for language, spec in LANGUAGES.items():
        seen: set[str] = set()
        for index in range(per_language):
            domain_index = index % len(DOMAINS)
            template = spec["templates"][domain_index]
            while True:
                values = {label: rng.choice(IPS if label == "IP" else spec[label]) for label in LABELS}
                code_switch = "LATIN_PERSON" in spec and rng.random() < CODE_SWITCH_SHARE
                if code_switch:
                    values["PERSON"] = rng.choice(spec["LATIN_PERSON"])
                text, entities = fill(template, values)
                if text not in seen:
                    break
            seen.add(text)
            records.append({
                "id": f"{language}-{DOMAINS[domain_index]}-{index + 1:03d}",
                "text": text,
                "language": language,
                "domain": DOMAINS[domain_index],
                "slices": ["synthetic", "multilingual"] + (["code-switch"] if code_switch else []),
                "entities": entities,
            })
    return records


def to_jsonl(records: list[dict]) -> str:
    return "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    args.output.write_text(to_jsonl(build(args.seed)), encoding="utf-8")


if __name__ == "__main__":
    main()
