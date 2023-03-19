import pymongo

client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["solidity"]

contracts_collection = db.get_collection("contracts")
libraries_collection = db.get_collection("libraries")
interfaces_collection = db.get_collection("interfaces")

print(contracts_collection.count_documents({}))
print(libraries_collection.count_documents({}))
print(interfaces_collection.count_documents({}))