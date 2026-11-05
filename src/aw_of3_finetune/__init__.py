"""AtomWorks-carried fine-tuning dataset cache for OpenFold3."""

import logging


class _HideUnsetMirrorNotice(logging.Filter):
    """Drop atomworks' import-time notice about unset mirrors; our commands take them as options."""

    def filter(self, record: logging.LogRecord) -> bool:
        return "MIRROR_PATH not set" not in record.getMessage()


logging.getLogger("atomworks.constants").addFilter(_HideUnsetMirrorNotice())
