create database user_warja;

use user_warja;

CREATE TABLE `users` (
    `user_id` VARCHAR(10) NOT NULL PRIMARY KEY,
    `email` VARCHAR(255) NOT NULL UNIQUE,
    `password` VARCHAR(255) NOT NULL,
    `name` VARCHAR(255) NOT NULL,
    `role` ENUM('admin', 'user', 'dosen') NOT NULL DEFAULT 'user',
    `profile` BLOB
);