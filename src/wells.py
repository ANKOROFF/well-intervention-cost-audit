"""Library: key normalization and date parsing used in joins."""
import polars as pl
CYR2LAT = [('В', 'V'), ('С', 'S'), ('А', 'A'), ('Е', 'E'), ('К', 'K'), ('М', 'M'), ('Н', 'N'), ('О', 'O'), ('Р', 'P'), ('Т', 'T'), ('Х', 'X'), ('У', 'Y')]
LAT2CYR = [(b, a) for a, b in CYR2LAT]
MONTHS = {'янв': 1, 'фев': 2, 'мар': 3, 'апр': 4, 'май': 5, 'июн': 6, 'июл': 7, 'авг': 8, 'сен': 9, 'окт': 10, 'ноя': 11, 'дек': 12}

def norm_well(col: pl.Expr) -> pl.Expr:
    c = col.cast(pl.Utf8).str.strip_chars().str.to_uppercase()
    for a, b in CYR2LAT:
        c = c.str.replace_all(a, b, literal=True)
    return c.str.replace_all('\\s+', '')

def norm_fleet(col: pl.Expr) -> pl.Expr:
    c = col.cast(pl.Utf8).str.strip_chars().str.replace('\\(.*$', '').str.replace_all('[\\s\\"]+', ' ').str.strip_chars().str.to_uppercase()
    for a, b in LAT2CYR:
        c = c.str.replace_all(a, b, literal=True)
    return c.str.replace_all('\\s*-\\s*', '-').str.strip_chars('- ')

def base_well(col: pl.Expr) -> pl.Expr:
    return col.str.replace('[_/-].*$', '')

def tr_month_end(year: pl.Expr, month: pl.Expr) -> pl.Expr:
    return pl.date(year, month, 1).dt.month_end()
