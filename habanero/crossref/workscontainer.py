from .collect import CollectResult, SourceResult


class WorksContainer:
    """
    WorksContainer: Class for working with works results

    :rtype: list

    Usage::

            from habanero import Crossref, WorksContainer
            cr = Crossref()

            res = cr.works(ids=['10.1136/jclinpath-2020-206745', '10.1136/esmoopen-2020-000776'])
            x = WorksContainer(res)
            x
            x.works
            x.doi
            x.license
            x.title
            x.abstract

            res2 = cr.works(limit = 2)
            x = WorksContainer(res2)
            x
            x.works
            x.doi
            x.license
            x.title
            x.abstract

            res3 = cr.members(ids = 98, works = True, limit = 5)
            x = WorksContainer(res3)
            x
            x.works
            x.doi
            x.license
            x.title
            x.abstract

    A :class:`~habanero.CollectResult` from :meth:`WorksQuery.collect` is also
    accepted, including an all-empty one (every source returned zero works).
    Works stay ordered by source input position, page, and item, and the
    source association is retained in parallel lists::

            result = WorksQuery(cr).members(["12", "34"]).collect()
            x = WorksContainer(result)
            x.works          # all works across both sources, ordered
            x.source_ids     # source ID per work, aligned with x.works
            x.source_pages   # 1-based page number per work
            x.source_indexes # input position of the source per work
            x.sources        # the SourceResult objects, in input order
    """

    def __init__(self, data) -> None:
        super().__init__()
        if isinstance(data, CollectResult):
            self.__data = data
            collect = True
            self.works = [work for source in data.sources for work in source.works]
        else:
            if not data:
                raise ValueError("data len must be > zero")
            self.__data = data
            collect = False
            self.works = self.works_handler(self.__data)
        keys = list(self.works[0].keys()) if self.works else []
        for key in keys:
            values = [work.get(key, None) for work in self.works]
            setattr(self, key.lower().replace("-", "_"), values)
        if collect:
            # set provenance last so a work field can never clobber it
            self.sources: list[SourceResult] = list(data.sources)
            self.source_indexes = [
                prov["source_index"]
                for source in data.sources
                for prov in source.provenance()
            ]
            self.source_ids = [
                prov["source_id"]
                for source in data.sources
                for prov in source.provenance()
            ]
            self.source_pages = [
                prov["page"] for source in data.sources for prov in source.provenance()
            ]
        else:
            self.sources = []

    def __repr__(self) -> str:
        return f"""<{type(self).__name__}: No. works: {len(self.works)}>"""

    def works_handler(self, x: list | dict) -> list:
        message_type = (
            next(w["message-type"] for w in x)
            if isinstance(x, list)
            else x["message-type"]
        )

        if isinstance(x, list):
            if x[0]["message-type"] == "work":
                x = list(filter(lambda w: w["message-type"] == "work", x))
                return [w["message"] for w in x]
            elif x[0]["message-type"] == "work-list":
                x = list(filter(lambda w: w["message-type"] == "work-list", x))
                items = [w["message"]["items"] for w in x]
                return [z for sublist in items for z in sublist]
            else:
                raise TypeError(
                    f"can only handle Crossref message-type 'work' & 'work-list', got: '{message_type}'"
                )
        elif isinstance(x, dict) and x["message-type"] == "work-list":
            return x["message"]["items"]
        elif isinstance(x, dict) and x["message-type"] == "work":
            return [x["message"]]
        else:
            raise TypeError(
                f"can only handle Crossref message-type 'work' & 'work-list', got: '{message_type}'"
            )
