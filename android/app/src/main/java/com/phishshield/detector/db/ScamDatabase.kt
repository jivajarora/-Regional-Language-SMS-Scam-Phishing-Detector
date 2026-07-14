package com.phishshield.detector.db

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase

@Database(entities = [ScamLog::class], version = 1, exportSchema = false)
abstract class ScamDatabase : RoomDatabase() {

    abstract fun scamLogDao(): ScamLogDao

    companion object {
        @Volatile
        private var INSTANCE: ScamDatabase? = null

        fun getDatabase(context: Context): ScamDatabase {
            return INSTANCE ?: synchronized(this) {
                val instance = Room.databaseBuilder(
                    context.applicationContext,
                    ScamDatabase::class.java,
                    "scam_detector_db"
                ).build()
                INSTANCE = instance
                instance
            }
        }
    }
}
