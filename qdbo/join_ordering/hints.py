import sqlparse
from sqlparse.sql import IdentifierList, Identifier
from sqlparse.tokens import Keyword


def build_table_alias_mapping(sql_query):
    try:
        parsed = sqlparse.parse(sql_query)[0]
        from_seen = False
        tables = []

        for token in parsed.tokens:
            if from_seen:
                if isinstance(token, IdentifierList):
                    for identifier in token.get_identifiers():
                        alias = identifier.get_alias()
                        table_name = identifier.get_real_name()
                        tables.append(alias if alias else table_name)

                elif isinstance(token, Identifier):
                    alias = token.get_alias()
                    table_name = token.get_real_name()
                    tables.append(alias if alias else table_name)

                elif token.ttype is Keyword and token.value.upper() in (
                    "WHERE",
                    "GROUP",
                    "ORDER",
                    "HAVING",
                    "LIMIT",
                ):
                    break

            elif token.ttype is Keyword and token.value.upper() == "FROM":
                from_seen = True

        mapping = {i: table for i, table in enumerate(tables) if table}
        return mapping

    except Exception as e:
        print(f"sqlparse failed: {e}")
        return {}


def build_postgres_hint(mapping, join_order):
    table_aliases = [mapping[int(i)] for i in join_order]
    hint = "/*+\n    Leading (" + " ".join(table_aliases) + ")\n*/"
    return hint


def prepend_hint_to_sql(sql_query, hint):
    return hint + "\n" + sql_query
