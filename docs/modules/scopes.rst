.. _scopes-modules:

=====================
ScopeCollection class
=====================

.. autoclass:: habanero.ScopeCollection
   :members:

.. autoclass:: habanero.MergedWorks
   :members:

.. autoclass:: habanero.MergedWork
   :members:

.. autoclass:: habanero.Source
   :members:

.. autoclass:: habanero.FieldVariant
   :members:

.. autoclass:: habanero.BranchFailure
   :members:

.. autodata:: habanero.MISSING

A short example merging a journal, a member and a funder scope::

    from habanero import Crossref, WorksQuery

    cr = Crossref()
    merged = WorksQuery.union(
        {
            "journal": WorksQuery(cr).journals("2167-8359").cursor("*"),
            "member": WorksQuery(cr).members(98).cursor("*"),
            "funder": WorksQuery(cr).funders("10.13039/100000001"),
        }
    ).execute()

    for work in merged:
        print(work.doi, work.scopes, len(work.sources))
        for variant in work.variants:
            print("  differs on", variant.field, "from", variant.source.branch)

    if merged.failures:
        for failure in merged.failures:
            print("branch failed:", failure.branch, failure.error_type, failure.message)
