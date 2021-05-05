from stock_prediction.db_model.wc_yahoo_finance import WCYahooFinance
from stock_prediction.db_model.news_entity import NewsEntity
from stock_prediction.db_model.entity import Entity
import functools
import re
import json

from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.after_request
def apply_caching(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response


@app.route('/v1/entityMapping', methods=["POST"])
@cross_origin()
def update_entity():
    content = request.json
    create = content['create']
    delete = content['delete']

    if len(create) != 0:
        for c in create:
            NewsEntity().insert_new_record(c)

    for d in delete:
        NewsEntity().mark_del_by_entity_id_and_idx(news_id=d['news_id'], entity_id=d['entity_id'],
                                                   str_idx=d['start_idx'], end_idx=d['end_idx'])

    return json.dumps({'success': True}), 200, {'ContentType': 'application/json'}


@app.route('/v1/stockCode', methods=["POST"])
@cross_origin()
def update_stock_code():
    content = request.json
    news_id = content['news_id']
    stock_code = content['stock_code']

    WCYahooFinance().update_stock_code_by_news_id(id=news_id, stock_code=stock_code)

    return json.dumps({'success': True}), 200, {'ContentType': 'application/json'}


@app.route('/v1/news/search', methods=["POST"])
@cross_origin()
def select_title_p_dt_by_criterion():
    content = request.json

    author = content.get('authors')
    stock_code = content.get('stock_codes')
    category = content.get('categories')
    source = content.get('sources')
    pub_str_date = content.get('published_start_date')
    pub_end_date = content.get('published_end_date')
    page = content.get('page')
    page_size = content.get('page_size')

    result, result_count = WCYahooFinance().select_title_p_dt_by_criterion(page, page_size, author=author,
                                                                           tag=stock_code,
                                                                           cat=category, src=source,
                                                                           post_str_dt=pub_str_date,
                                                                           post_end_dt=pub_end_date)

    return jsonify({
        "result": [
            {
                "id": row['ID'],
                "title": row['TITLE'],
                "publication_date": row['POST_DT'],
                "update_date": row['UP_DT']
            }
            for row in result.fetchall()
        ],
        "result_count": result_count.fetchone()[0]
    }
    )


@app.route('/v1/details/<int:news_id>')
def select_news_by_id(news_id):
    result = {}
    news_content = WCYahooFinance().select_news_by_id(id=news_id).fetchone()

    result['news'] = {
        'id': news_id,
        'title': news_content['TITLE'],
        'publication_date': news_content['POST_DT'],
        'update_date': news_content['UP_DT'],
        'url': news_content['LINK'],
        'domain': 'http://finance.yahoo.com',
        'content': news_content['CONTENT'],
        'summary': news_content['SUMMARY'],
        'tag': news_content['TAG'],
        'source': news_content['SRC'],
        'next_news_id': WCYahooFinance().get_next_id(news_id),
    }

    entity_color = Entity().select_entity_color().fetchall()

    entity_color_dict = {}
    for ec in entity_color:
        entity_color_dict[ec['ID']] = ec['HIGHLIGHT_COLOR']

    result['entity'] = [{
        'id': row['ID'],
        'name': row['NAME'],
        'bg_color': row['HIGHLIGHT_COLOR']
    }
        for row in entity_color
    ]

    result['highlighted_text'] = [
        {
            'start_idx': row['START_IDX'],
            'end_idx': row['END_IDX'],
            'bg_color': entity_color_dict.get(row['ENTITY_ID']),
            'entity_id': row['ENTITY_ID'],
            'entity_value': row['ENTITY_VALUE']
        }
        for row in NewsEntity().select_entity_by_news_id(news_id).fetchall()
    ]

    return jsonify(result)


@app.route('/v1/news/refinements')
def select_author_category_stockcode():
    author = list(
        set(functools.reduce(lambda x, y: x + y, [[j.strip() for j in re.split(',| and ', i[0]) if j != ''] for i in
                                                  WCYahooFinance().select_distinct_author().fetchall()
                                                  if i[0] not in (
                                                      'Editor focused on markets and the economy',
                                                      None)])))
    category = [i[0] for i in WCYahooFinance().select_distinct_category().fetchall() if i[0] is not None]
    stock_code = list(set(functools.reduce(lambda x, y: x + y,
                                           [i[0].split(',') for i in WCYahooFinance().select_distinct_tag().fetchall()
                                            if i[0] is not None])))

    author.sort()
    category.sort()
    stock_code.sort()
    return jsonify({
        'authors': author,
        'categories': category,
        'stockCode': stock_code
    })
