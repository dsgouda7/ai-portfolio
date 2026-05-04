#!/usr/bin/env python3
"""A/B test traffic splitting controller"""

import random
import sqlite3
from datetime import datetime

class ABTestController:
    def __init__(self, db_path='ab_test.db'):
        self.conn = sqlite3.connect(db_path)
        self._init_db()
    
    def _init_db(self):
        """Initialize AB test configuration table"""
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS ab_config (
                model_version TEXT PRIMARY KEY,
                traffic_percentage REAL,
                enabled INTEGER
            )
        ''')
        self.conn.commit()
    
    def set_traffic(self, model_version, percentage):
        """Set traffic percentage for a model version"""
        self.conn.execute('''
            INSERT OR REPLACE INTO ab_config (model_version, traffic_percentage, enabled)
            VALUES (?, ?, 1)
        ''', (model_version, percentage))
        self.conn.commit()
        print(f"✅ Set {model_version} to {percentage}% traffic")
    
    def route_request(self, request_id):
        """Determine which model version to route a request to"""
        cursor = self.conn.execute('''
            SELECT model_version, traffic_percentage
            FROM ab_config WHERE enabled = 1
            ORDER BY traffic_percentage DESC
        ''')
        
        versions = cursor.fetchall()
        rand = random.random() * 100
        cumulative = 0
        
        for version, percentage in versions:
            cumulative += percentage
            if rand < cumulative:
                return version
        
        return versions[0][0] if versions else 'v1'

if __name__ == "__main__":
    controller = ABTestController()
    controller.set_traffic('v1', 90.0)
    controller.set_traffic('v2', 10.0)
    print("A/B test controller configured!")
