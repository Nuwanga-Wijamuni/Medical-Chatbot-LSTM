from flask import Flask, request, jsonify, render_template, redirect
from flask_cors import CORS
import mysql.connector
from util import Util
from flask_socketio import SocketIO, emit
from brain import run_chat_bot

app = Flask(__name__)

CORS(app)

DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'db': ' sliit_ml_chat',
}
socketio = SocketIO(app, cors_allowed_origins="http://localhost:3000")
utility = Util()


def is_username_exist(username):
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        with connection.cursor(buffered=True) as cursor:
            sql = "SELECT * FROM users WHERE username=%s"
            cursor.execute(sql, (username,))
            connection.close()
            if cursor.rowcount > 0:
                return True
            else:
                return False
    except Exception as e:
        print(e)
        return False


@app.route('/api/register', methods=['POST'])
def start_chat():
    data = request.get_json()
    if is_username_exist(data['username']):
        return utility.response('error', 'Username already exist', 'user_exists')
    connection = mysql.connector.connect(**DB_CONFIG)
    try:
        token = utility.generate_random_string(50)
        with connection.cursor(buffered=True, dictionary=True) as cursor:
            sql = "INSERT INTO users(username, password, token) VALUES(%s, %s, %s)"
            cursor.execute(sql, (
                data['username'],
                utility.hash_password(data['password']),
                token
            ))
            connection.commit()
        connection.close()
        return utility.response('success', 'User profile created', 'success')
    except Exception as e:
        return utility.response('error', 'Something went wrong', 'db_error')


@app.route('/api/login', methods=["POST"])
def login():
    data = request.get_json()
    try:
        connection = mysql.connector.connect(**DB_CONFIG)

        if is_username_exist(data['username']):
            sql = "SELECT * FROM users WHERE username=%s"
            with connection.cursor(buffered=True, dictionary=True) as cursor:
                cursor.execute(sql, (data['username'],))
                row = cursor.fetchone()
                print(utility.verify_password(data['password'], row['password']))
                if utility.verify_password(data['password'], row['password']):
                    return utility.response('success', 'success', 'success', {
                        'token': row['token'],
                        'id': row['id']
                    })
                return utility.response('error', 'Username or password incorrect', 'auth_error')
        return utility.response('error', 'Username does not exist', 'user_does_not_exists')
    except Exception as e:
        print(e)
        return utility.response('error', 'Something went wrong', 'db_error')


@app.route('/api/get-messages', methods=["POST"])
def get_messages():
    data = request.get_json()
    connection = mysql.connector.connect(**DB_CONFIG)
    with connection.cursor(buffered=True, dictionary=True) as cursor:
        sql = "SELECT * FROM chats WHERE user_token=%s ORDER BY sentOn ASC"
        cursor.execute(sql, (data['id'],))
        rows = cursor.fetchall()
        connection.close()
        return utility.response('success', 'success', 'success', {
            'messages': rows
        })

@socketio.on('message')
def handle_message(message):
    client_sid = request.sid
    print(message['message'])
    connection = mysql.connector.connect(**DB_CONFIG)
    with connection.cursor(buffered=True) as cursor:
        sql = "INSERT INTO chats(user_token, isUser, message) VALUES(%s, %s, %s)"
        cursor.execute(sql, (message['id'], 1, message['message']))
        connection.commit()
        cursor.close()
    reply = run_chat_bot(message['message'])
    with connection.cursor(buffered=True) as cursor:
        sql = "INSERT INTO chats(user_token, isUser, message) VALUES(%s, %s, %s)"
        cursor.execute(sql, (message['id'], 0, reply))
        connection.commit()
        cursor.close()
    connection.close()
    emit('message', reply, room=client_sid)


if __name__ == '__main__':
    socketio.run(app, debug=True, port=5000)
