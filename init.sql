CREATE TABLE IF NOT EXISTS users (
    userid VARCHAR(100) PRIMARY KEY,
    passwd VARCHAR(100) NOT NULL,
    nickname VARCHAR(100),
    profile_url VARCHAR(300),
    is_online BOOLEAN,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS musics (
    musicid VARCHAR(255) NOT NULL PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    artist VARCHAR(100),
    accompaniment_path VARCHAR(300),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS music_meta (
    musicid VARCHAR(255) NOT NULL PRIMARY KEY,
    pitch_vector FLOAT8[],
    onset_times FLOAT8[]
);