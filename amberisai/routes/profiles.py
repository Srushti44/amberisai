from flask import Blueprint, request, jsonify
from models.database import create_baby, get_baby

profiles_bp = Blueprint('profiles', __name__)


@profiles_bp.route('/profile', methods=['POST'])
def create_profile():
    data = request.get_json()

    if not data:
        return jsonify({"success": False, "error": "JSON body required"}), 400

    nickname = data.get('nickname')
    if not nickname:
        return jsonify({"success": False, "error": "nickname is required"}), 400

    age_days = data.get('age_days', None)
    allergies = data.get('allergies', [])

    if not isinstance(allergies, list):
        return jsonify({"success": False, "error": "allergies must be a list"}), 400

    baby_id = create_baby(nickname, age_days, allergies)

    return jsonify({
        "success": True,
        "baby_id": baby_id,
        "nickname": nickname,
        "age_days": age_days,
        "allergies": allergies
    }), 201


@profiles_bp.route('/profile/<int:baby_id>', methods=['GET'])
def get_profile(baby_id):
    baby = get_baby(baby_id)
    if not baby:
        return jsonify({"success": False, "error": f"Baby with id {baby_id} not found"}), 404

    return jsonify({"success": True, "baby": baby}), 200