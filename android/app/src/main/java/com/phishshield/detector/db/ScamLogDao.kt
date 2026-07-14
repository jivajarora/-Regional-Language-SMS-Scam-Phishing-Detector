package com.phishshield.detector.db

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import kotlinx.coroutines.flow.Flow

@Dao
interface ScamLogDao {

    @Query("SELECT * FROM scam_logs ORDER BY timestamp DESC")
    fun getAllLogs(): Flow<List<ScamLog>>

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertLog(log: ScamLog)

    @Query("UPDATE scam_logs SET userFeedback = :feedback WHERE id = :id")
    suspend fun updateFeedback(id: Int, feedback: String)

    @Query("DELETE FROM scam_logs WHERE id = :id")
    suspend fun deleteLog(id: Int)

    @Query("DELETE FROM scam_logs")
    suspend fun clearAll()
}
