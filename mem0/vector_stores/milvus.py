import logging
from typing import Dict, Optional

from pydantic import BaseModel

from mem0.configs.vector_stores.milvus import MetricType
from mem0.vector_stores.base import VectorStoreBase

try:
    import pymilvus  # noqa: F401
except ImportError:
    raise ImportError("The 'pymilvus' library is required. Please install it using 'pip install pymilvus'.")

from pymilvus import CollectionSchema, DataType, FieldSchema, MilvusClient

logger = logging.getLogger(__name__)


class OutputData(BaseModel):
    id: Optional[str]  # memory id
    score: Optional[float]  # distance
    payload: Optional[Dict]  # metadata


class MilvusDB(VectorStoreBase):
    def __init__(
        self,
        url: str,
        token: str,
        collection_name: str,
        embedding_model_dims: int,
        metric_type: MetricType,
        db_name: str,
    ) -> None:
        """Initialize the MilvusDB database.

        Args:
            url (str): Full URL for Milvus/Zilliz server.
            token (str): Token/api_key for Zilliz server / for local setup defaults to None.
            collection_name (str): Name of the collection (defaults to mem0).
            embedding_model_dims (int): Dimensions of the embedding model (defaults to 1536).
            metric_type (MetricType): Metric type for similarity search (defaults to L2).
            db_name (str): Name of the database (defaults to "").
        """
        self.collection_name = collection_name
        self.embedding_model_dims = embedding_model_dims
        self.metric_type = metric_type
        self.client = MilvusClient(uri=url, token=token, db_name=db_name)
        self.create_col(
            collection_name=self.collection_name,
            vector_size=self.embedding_model_dims,
            metric_type=self.metric_type,
        )

    def create_col(
        self,
        collection_name: str,
        vector_size: int,
        metric_type: MetricType = MetricType.COSINE,
    ) -> None:
        """Create a new collection with index_type AUTOINDEX.

        Args:
            collection_name (str): Name of the collection (defaults to mem0).
            vector_size (int): Dimensions of the embedding model (defaults to 1536).
            metric_type (MetricType, optional): etric type for similarity search. Defaults to MetricType.COSINE.
        """

        if self.client.has_collection(collection_name):
            logger.info(f"Collection {collection_name} already exists. Skipping creation.")
            # Ensure collection is loaded into memory for search operations
            load_state = self.client.get_load_state(collection_name=collection_name)
            if load_state.get("state") != "Loaded":
                logger.info(f"Loading collection {collection_name} into memory.")
                self.client.load_collection(collection_name=collection_name)
        else:
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=512),
                FieldSchema(name="vectors", dtype=DataType.FLOAT_VECTOR, dim=vector_size),
                FieldSchema(name="metadata", dtype=DataType.JSON),
            ]

            schema = CollectionSchema(fields, enable_dynamic_field=True)

            # Create index parameters for vector field and JSON metadata fields
            index_params = self.client.prepare_index_params()

            # Vector index for similarity search
            index_params.add_index(
                field_name="vectors",
                metric_type=metric_type,
                index_type="AUTOINDEX",
                index_name="vector_index"
            )

            # JSON path indexes for faster metadata filtering (Milvus 2.5.11+)
            # Index user_id for user-scoped queries
            index_params.add_index(
                field_name="metadata",
                index_type="INVERTED",
                index_name="user_id_index",
                params={
                    "json_path": 'metadata["user_id"]',
                    "json_cast_type": "varchar"
                }
            )

            # Index event_dates array for date-based filtering
            index_params.add_index(
                field_name="metadata",
                index_type="INVERTED",
                index_name="event_dates_index",
                params={
                    "json_path": 'metadata["event_dates"]',
                    "json_cast_type": "array_varchar"
                }
            )

            # Index entity_names array for entity-based filtering
            index_params.add_index(
                field_name="metadata",
                index_type="INVERTED",
                index_name="entity_names_index",
                params={
                    "json_path": 'metadata["entity_names"]',
                    "json_cast_type": "array_varchar"
                }
            )

            self.client.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)

    def insert(self, ids, vectors, payloads, **kwargs: Optional[dict[str, any]]):
        """Insert vectors into a collection.

        Args:
            vectors (List[List[float]]): List of vectors to insert.
            payloads (List[Dict], optional): List of payloads corresponding to vectors.
            ids (List[str], optional): List of IDs corresponding to vectors.
        """
        # Batch insert all records at once for better performance and consistency
        data = [
            {"id": idx, "vectors": embedding, "metadata": metadata}
            for idx, embedding, metadata in zip(ids, vectors, payloads)
        ]
        self.client.insert(collection_name=self.collection_name, data=data, **kwargs)

    def _create_filter(self, filters: dict):
        """Prepare filters for efficient query.

        Args:
            filters (dict): filters [user_id, agent_id, run_id] or advanced operators like:
                - {"field": {"in": [values]}} for json_contains_any
                - {"field": {"contains": value}} for json_contains
                - {"field": {"all": [values]}} for json_contains_all

        Returns:
            str: formatted filter expression.
        """
        operands = []
        for key, value in filters.items():
            if isinstance(value, dict):
                # Handle advanced operators for entity filtering
                if "in" in value:
                    # json_contains_any - check if field contains ANY of the values
                    values_list = value["in"]
                    if isinstance(values_list, list):
                        formatted_values = [f'"{v}"' if isinstance(v, str) else str(v) for v in values_list]
                        operands.append(f'json_contains_any(metadata["{key}"], [{", ".join(formatted_values)}])')
                elif "contains" in value:
                    # json_contains - check if field contains a single value
                    v = value["contains"]
                    if isinstance(v, str):
                        operands.append(f'json_contains(metadata["{key}"], "{v}")')
                    else:
                        operands.append(f'json_contains(metadata["{key}"], {v})')
                elif "all" in value:
                    # json_contains_all - check if field contains ALL values
                    values_list = value["all"]
                    if isinstance(values_list, list):
                        formatted_values = [f'"{v}"' if isinstance(v, str) else str(v) for v in values_list]
                        operands.append(f'json_contains_all(metadata["{key}"], [{", ".join(formatted_values)}])')
                elif "gte" in value or "lte" in value or "gt" in value or "lt" in value:
                    # Range operators for date filtering
                    for op, op_val in value.items():
                        op_map = {"gte": ">=", "lte": "<=", "gt": ">", "lt": "<"}
                        if op in op_map:
                            if isinstance(op_val, str):
                                operands.append(f'(metadata["{key}"] {op_map[op]} "{op_val}")')
                            else:
                                operands.append(f'(metadata["{key}"] {op_map[op]} {op_val})')
            elif isinstance(value, str):
                operands.append(f'(metadata["{key}"] == "{value}")')
            else:
                operands.append(f'(metadata["{key}"] == {value})')

        return " and ".join(operands)

    def _parse_output(self, data: list):
        """
        Parse the output data.

        Args:
            data (Dict): Output data.

        Returns:
            List[OutputData]: Parsed output data.
        """
        memory = []

        for value in data:
            uid, score, metadata = (
                value.get("id"),
                value.get("distance"),
                value.get("entity", {}).get("metadata"),
            )

            memory_obj = OutputData(id=uid, score=score, payload=metadata)
            memory.append(memory_obj)

        return memory

    def search(self, query: str, vectors: list, limit: int = 5, filters: dict = None) -> list:
        """
        Search for similar vectors.

        Args:
            query (str): Query.
            vectors (List[float]): Query vector.
            limit (int, optional): Number of results to return. Defaults to 5.
            filters (Dict, optional): Filters to apply to the search. Defaults to None.

        Returns:
            list: Search results.
        """
        query_filter = self._create_filter(filters) if filters else None
        hits = self.client.search(
            collection_name=self.collection_name,
            data=[vectors],
            anns_field="vectors",
            limit=limit,
            filter=query_filter,
            output_fields=["*"],
        )
        result = self._parse_output(data=hits[0])
        return result

    def delete(self, vector_id):
        """
        Delete a vector by ID.

        Args:
            vector_id (str): ID of the vector to delete.
        """
        self.client.delete(collection_name=self.collection_name, ids=vector_id)

    def update(self, vector_id=None, vector=None, payload=None):
        """
        Update a vector and its payload.

        Args:
            vector_id (str): ID of the vector to update.
            vector (List[float], optional): Updated vector.
            payload (Dict, optional): Updated payload.
        """
        schema = {"id": vector_id, "vectors": vector, "metadata": payload}
        self.client.upsert(collection_name=self.collection_name, data=schema)

    def get(self, vector_id):
        """
        Retrieve a vector by ID.

        Args:
            vector_id (str): ID of the vector to retrieve.

        Returns:
            OutputData: Retrieved vector.
        """
        result = self.client.get(collection_name=self.collection_name, ids=vector_id)
        output = OutputData(
            id=result[0].get("id", None),
            score=None,
            payload=result[0].get("metadata", None),
        )
        return output

    def list_cols(self):
        """
        List all collections.

        Returns:
            List[str]: List of collection names.
        """
        return self.client.list_collections()

    def delete_col(self):
        """Delete a collection."""
        return self.client.drop_collection(collection_name=self.collection_name)

    def col_info(self):
        """
        Get information about a collection.

        Returns:
            Dict[str, Any]: Collection information.
        """
        return self.client.get_collection_stats(collection_name=self.collection_name)

    def list(self, filters: dict = None, limit: int = 100) -> list:
        """
        List all vectors in a collection.

        Args:
            filters (Dict, optional): Filters to apply to the list.
            limit (int, optional): Number of vectors to return. Defaults to 100.

        Returns:
            List[OutputData]: List of vectors.
        """
        query_filter = self._create_filter(filters) if filters else None
        result = self.client.query(collection_name=self.collection_name, filter=query_filter, limit=limit)
        memories = []
        for data in result:
            obj = OutputData(id=data.get("id"), score=None, payload=data.get("metadata"))
            memories.append(obj)
        return [memories]

    def reset(self):
        """Reset the index by deleting and recreating it."""
        logger.warning(f"Resetting index {self.collection_name}...")
        self.delete_col()
        self.create_col(self.collection_name, self.embedding_model_dims, self.metric_type)
