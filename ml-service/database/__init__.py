"""
Database Module - Kết nối với MongoDB
======================================
Module này cung cấp kết nối đến MongoDB và các thao tác CRUD
cho users, transactions, predictions
"""

from .mongodb import MongoDBConnection, get_db

__all__ = ['MongoDBConnection', 'get_db']
