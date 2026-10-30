import ipaddress

from sct.utils import constants


def is_replaceable_ip(candidate, public_only=False):
    """True if ``candidate`` is a valid IP (and public, when ``public_only``)."""
    try:
        ip = ipaddress.ip_address(candidate)
    except ValueError:
        return False
    return ip.is_global or not public_only


class ProcessContacts:

    def __init__(self):
        pass

    def replace_urls(self, text, replace_with="<URL>"):
        """Replace all URLs in ``text`` str with ``replace_with`` str."""
        return constants.URL_REGEX.sub(replace_with, text)

    def replace_html(self, text, replace_with="<HTML>"):
        """Replace all html tags in ``text`` str with ``replace_with`` str."""
        return constants.HTML_REGEX.sub(replace_with, text)

    def replace_emails(self, text, replace_with="<EMAIL>"):
        """Replace all emails in ``text`` str with ``replace_with`` str."""
        return constants.EMAIL_REGEX.sub(replace_with, text)

    def replace_ips(self, text, replace_with="<IP>", public_only=False):
        """Replace IPv4/IPv6 addresses in ``text`` (only public ones if ``public_only``)."""
        return self.replace_ips_with_fn(text, lambda orig: replace_with, public_only)

    def replace_phone_numbers(self, text, replace_with="<PHONE>"):
        """Replace all phone numbers in ``text`` str with ``replace_with`` str."""
        return constants.PHONE_REGEX.sub(replace_with, text)

    def replace_numbers(self, text, replace_with="<NUMBER>"):
        """Replace all numbers in ``text`` str with ``replace_with`` str."""
        return constants.NUMBERS_REGEX.sub(replace_with, text)

    # --- Callable-based replacements (for synthetic mode) ---

    def replace_emails_with_fn(self, text, replace_fn):
        """Replace emails using a callable: replace_fn(matched_text) -> replacement."""
        return constants.EMAIL_REGEX.sub(lambda m: replace_fn(m.group()), text)

    def replace_urls_with_fn(self, text, replace_fn):
        """Replace URLs using a callable: replace_fn(matched_text) -> replacement."""
        return constants.URL_REGEX.sub(lambda m: replace_fn(m.group()), text)

    def replace_ips_with_fn(self, text, replace_fn, public_only=False):
        """Replace IPs using a callable: replace_fn(matched_text) -> replacement."""
        return constants.IP_REGEX.sub(
            lambda m: replace_fn(m.group()) if is_replaceable_ip(m.group(), public_only) else m.group(),
            text,
        )

    def replace_phone_numbers_with_fn(self, text, replace_fn):
        """Replace phone numbers using a callable: replace_fn(matched_text) -> replacement."""
        return constants.PHONE_REGEX.sub(lambda m: replace_fn(m.group()), text)

    def replace_numbers_with_fn(self, text, replace_fn):
        """Replace numbers using a callable: replace_fn(matched_text) -> replacement."""
        return constants.NUMBERS_REGEX.sub(lambda m: replace_fn(m.group()), text)
