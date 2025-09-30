{% set title = '.'.join(fullname.split('.')[-2:]) %}
{{ title }}
{{ underline }}

.. currentmodule:: {{ module }}

.. autoattribute:: {{ objname }}