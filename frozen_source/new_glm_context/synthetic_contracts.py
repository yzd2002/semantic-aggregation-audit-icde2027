"""Original synthetic records, copied without changing the features or order."""


def synthetic_records(task, context):
    if task == 'dblp_acm':
        a = {'title': 'Database systems', 'authors': 'Alice Smith', 'venue': 'Example Conference', 'year': '2020'}
        b = {'title': 'Cloud formations', 'authors': 'Bob Jones', 'venue': 'Weather Journal', 'year': '2019'}
        c = {'title': 'Graph algorithms', 'authors': 'Carol Brown', 'venue': 'Computing Journal', 'year': '2018'}
        pairs = [(a, a), (b, b), (b, c), (c, c)] if context == 'near' else [(a, a), (c, b), (a, c), (b, b)]
        return [{'features': {'left': x, 'right': y}} for x, y in pairs]
    target = {'question': 'Is water a liquid at room temperature?', 'passage': 'At ordinary room temperature and pressure, water is a liquid.'}
    others = [{'question': 'Can a triangle have four sides?', 'passage': 'A triangle has exactly three sides.'},
              {'question': 'Is the Moon a star?', 'passage': 'The Moon is a natural satellite, not a star.'},
              {'question': 'Do birds have feathers?', 'passage': 'Birds have feathers.'}]
    if context == 'random':
        others = list(reversed(others))
    return [{'features': r} for r in [target, *others]]
