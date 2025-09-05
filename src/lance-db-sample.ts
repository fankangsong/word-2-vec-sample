import * as lancedb from "@lancedb/lancedb";
import { Schema, Field, Utf8, FixedSizeList, Float32 } from "apache-arrow";

const fs = require("fs");
const path = require("path");

// 预训练 GloVe 文件路径（请修改为你解压后的文件路径）
const GLOVE_FILE = path.join(__dirname, "./models/glove.6B.50d.txt");

// 加载 GloVe 模型
const gloveModel = loadGlove(GLOVE_FILE);

// 用来测试的单词列表
const wordsToCheck = [
  "cat",
  "dog",
  "kitten",
  "puppy",
  "apple",
  "banana",
  "orange",
  "fruit",
  "car",
  "bus",
  "train",
  "vehicle",
  "king",
  "queen",
  "man",
  "woman",
];

// 定义数据库文件目录
const DB_DIR = "./lancedb_data";
const TABLE_NAME = "vector_table";

const VECTOR_DIM = 50; // 向量维度

// 定义LanceDB表的Schema
const DB_SCHEMA = new Schema([
  new Field("word", new Utf8(), false),
  new Field(
    "vector",
    new FixedSizeList(VECTOR_DIM, new Field("item", new Float32(), false)),
    false
  ),
]);

// 获取LanceDB表的实例对象
async function getLanceDBTable() {
  try {
    const db = await lancedb.connect(DB_DIR);
    const tables = await db.tableNames();

    if (!tables.includes(TABLE_NAME)) {
      await db.createTable(TABLE_NAME, [], {
        schema: DB_SCHEMA,
      });
    }

    return db.openTable(TABLE_NAME);
  } catch (error) {
    console.error("Error accessing LanceDB:", error);
    throw error;
  }
}

interface VectorEntry {
  word: string;
  vector: number[];
}

// 写入数据库
async function insertVec(vectorEntry: VectorEntry[]) {
  const table = await getLanceDBTable();

  const dataToInsert = vectorEntry.map((entry) => ({
    word: entry.word,
    vector: entry.vector,
  }));

  await table.add(dataToInsert);
}

async function retrieve(vector: number[], topK: number = 5) {
  const table = await getLanceDBTable();
  const results = await table.search(vector, "vector").limit(topK).toArray();
  return results;
}

// 读取模型
function loadGlove(filePath: string) {
  console.log("正在加载 GloVe 模型，请稍候...");

  const lines = fs.readFileSync(filePath, "utf8").split("\n");
  const model: Record<string, number[]> = {};
  for (let line of lines) {
    if (!line) continue;
    const parts = line.split(" ");
    const word = parts[0];
    const vector = parts.slice(1).map(Number);
    model[word] = vector;
  }
  console.log("✅ 模型加载完成");
  return model;
}

function word2vec() {
  const vectorEntries: VectorEntry[] = [];

  // 开始向量化
  wordsToCheck.forEach((word) => {
    if (gloveModel[word]) {
      console.log(
        `${word}: [${gloveModel[word]
          .slice(0, 10)
          .map((v) => v.toFixed(3))
          .join(", ")} ...]`
      );
      vectorEntries.push({
        word,
        vector: gloveModel[word].slice(0, VECTOR_DIM),
      });
    } else {
      console.log(`${word}: ❌ 不在 GloVe 模型词表中`);
    }
  });

  return vectorEntries;
}

// 相似度查询
async function retrieveSimilarWords() {
  wordsToCheck.forEach(async (word) => {
    const wordVec = gloveModel[word].slice(0, VECTOR_DIM);
    console.log(`\n与 "${word}" 最相似的词有：`);
    const results = await retrieve(wordVec, 3);

    console.log('--------------------------')
    console.log('与', word, '最相似的词有：')
    console.log(results.map((item: any) => item.word))
    console.log('--------------------------')
  });
}

async function main() {
  // 获取向量化结果
  const vectorEntries = word2vec();

  // 插入到 LanceDB
  await insertVec(vectorEntries);

  retrieveSimilarWords();
}

main();
