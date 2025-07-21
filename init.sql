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
    title VARCHAR(255),
    artist VARCHAR(100),
    genre VARCHAR(100),
    accompaniment_path VARCHAR(300),
    created_at TIMESTAMP DEFAULT NOW()
);



CREATE TABLE IF NOT EXISTS music_meta (
    musicid VARCHAR(255) NOT NULL PRIMARY KEY,
    pitch_vector FLOAT8[],
    onset_times FLOAT8[],
    lyrics TEXT
);

CREATE TABLE IF NOT EXISTS challenges (
    challengeid SERIAL NOT NULL PRIMARY KEY,
    title VARCHAR(255),
    descript VARCHAR(500)
);

CREATE TABLE IF NOT EXISTS user_challenges(
    usrchalid SERIAL PRIMARY KEY,
    userid VARCHAR(100),
    challengeid SERIAL NOT NULL
);

CREATE TABLE IF NOT EXISTS user_records(
    recordid SERIAL PRIMARY KEY,
    userid VARCHAR(100),
    musicid VARCHAR(255),
    score FLOAT,
    audio_url VARCHAR(300),
    pitch_vector FLOAT8[],
    onset_times FLOAT8[],
    created_at TIMESTAMP DEFAULT NOW()
);