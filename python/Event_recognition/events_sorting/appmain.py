import flask
from flask import request
import triples_compare as em

webApp = flask.Flask(__name__)
webApp.config.update(RESTFUL_JSON=dict(ensure_ascii=False))


@webApp.route("/ematch", methods=['GET', 'POST'])
def event_match():
    method=request.method
    print(f"method={method}")
    new_content = request.form.get("content")
    val = em.execute(new_content)
    print(f"val={val}")
    return val


@webApp.route("/test")
def test():
    content = "朝鲜当地时间凌晨2时59分许和3时23分许在咸镜南道永兴一带向东部海域发射不明飞行器"
    val = em.execute(content)
    print(f"val={val}")
    return val


if __name__ == '__main__':
    webApp.config['JSON_AS_ASCII'] = False
    
    webApp.run(host='0.0.0.0', port=38080, debug=True)
