import { MongoClient, Db, Collection } from "mongodb";
import { createHash } from "crypto";

export interface SolidityFunction {
  declaration: string;
  body: string;
  hash?: string;
}

export interface SolidityContract {
  name: string;
  source_code: string;
  md5_hash: string;
}

const uri = "mongodb://localhost:27017";
const dbName = "solidity";

export class Connection {
  private client: MongoClient;
  private db?: Db;
  public functions_collection?: Collection;
  public contracts_collection?: Collection;
  public libraries_collection?: Collection;

  constructor() {
    this.client = new MongoClient(uri);
  }

  async init() {
    await this.client.connect();
    this.db = this.client.db(dbName);
    this.functions_collection = this.db.collection("functions");
    this.contracts_collection = this.db.collection("contracts");
    this.libraries_collection = this.db.collection("libraries");
  }

  public async saveFunction(func: SolidityFunction) {
    if (!this.functions_collection) {
      await this.init();
    }
    if (!func.hash) {
      const hash = createHash("md5");
      hash.update(func.declaration + func.body);
      return hash.digest("hex");
    }
    try {
      await this.functions_collection!.insertOne(func);
    } catch (err: any) {
      console.log(err);
    }
  }

  public async getContracts(skip: number, limit: number, filter = {}): Promise<SolidityContract[]> {
    if (!this.contracts_collection) {
      await this.init();
    }
    let cursor = await this.contracts_collection!.find(filter, { skip, limit });
    let res: any[] = [];
    await cursor.forEach((x) => {
      res.push(x);
    });
    return res as SolidityContract[];
  }
}

export const connection = new Connection();
