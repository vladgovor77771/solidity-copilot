import * as parser from "solidity-parser-antlr";
import { SolidityFunction, connection } from "./db";

function solidityToAst(sourceCode: string) {
  try {
    const ast = parser.parse(sourceCode, { loc: true, range: true });
    return ast;
  } catch (error) {
    console.error(`Error parsing Solidity code: ${error}`);
    return null;
  }
}

function extractFunctions(ast: any, full_source: string): SolidityFunction[] {
  const functions: SolidityFunction[] = [];

  function traverse(node: any) {
    if (
      node.type == "FunctionDefinition" ||
      node.type == "ModifierDefinition"
    ) {
      let funcSource = full_source.slice(node.range[0], node.range[1] + 1);
      let bodyStartIndex = funcSource.indexOf("{");
      let declaration = funcSource
        .substring(0, bodyStartIndex)
        .trim()
        .replace(/\n/g, " ")
        .replace(/ +/g, " ");
      let body = funcSource.substring(bodyStartIndex);
      functions.push({
        declaration,
        body,
      });
    }

    for (const key in node) {
      const value = node[key];
      if (typeof value === "object") {
        if (Array.isArray(value)) {
          for (const item of value) {
            if (typeof item === "object" && item !== null) {
              traverse(item);
            }
          }
        } else if (value !== null) {
          traverse(value);
        }
      }
    }
  }

  if (ast) {
    traverse(ast);
  }

  return functions;
}

async function main() {
  await connection.init();

  let totalContracts = await connection.contracts_collection!.countDocuments();
  for (let i = 2; i < totalContracts; ++i) {
    console.log(i)
    let contracts = await connection.getContracts(i, 1);
    console.log(contracts[0].source_code)
    let ast = solidityToAst(contracts[0].source_code);
    let functions = extractFunctions(ast, contracts[0].source_code);
    for (let func of functions) {
      await connection.saveFunction(func);
    }
    break;
    // if (i % 100 == 0) {
    //   console.log(i);
    // }
  }
}

main();
