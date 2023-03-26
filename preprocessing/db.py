import pymongo

client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["solidity"]

contracts_collection = db.get_collection("contracts")
libraries_collection = db.get_collection("libraries")
interfaces_collection = db.get_collection("interfaces")
functions_collections = db.get_collection("functions")
# print(functions_collections.delete_many({ "declaration": { "$regex": ";" }}))
# print(functions_collections.count_documents({}))