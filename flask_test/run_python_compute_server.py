from flask import Flask, jsonify, request

app = Flask(__name__)


def dummy_function(num1: int, num2: int) -> int:
    """Add some ints"""
    if not isinstance(num1, int) or not isinstance(num2, int):
        raise TypeError("Both arguments must be integers.")
    return num1 + num2


@app.route("/run_code", methods=["POST"])
def run_code():
    input_data = request.json
    output_data = dummy_function(
        **input_data
    )  # Unpack the input data as keyword arguments
    result = {"input": input_data, "output": output_data}  # Response
    response = jsonify(result)
    response.headers["Content-Type"] = "application/json"
    return response


if __name__ == "__main__":
    app.run(debug=True)
